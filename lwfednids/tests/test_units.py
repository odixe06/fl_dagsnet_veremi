"""Fast CPU unit checks: parameter counts (unpruned and pruned), the pruning plan
(deterministic, rebuilds torch-pruning's model exactly, keeps the L1-largest channels,
survives a JSON round trip, rejects a wrong model), flat round-trip, BN folding exactness
on the PRUNED model, the CE objective, the client update, the 1/N aggregate of Eq. (9),
the LR schedule, and the 10 metrics against sklearn."""
import copy, json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/lwfednids-bouayad-2024"))
from proj.model import build_model, build_full, CFG as MCFG, N_PARAMS_FULL, n_params
from proj.prune import compute_plan, apply_plan, plan_id, PRUNABLE
from proj.lwfednids import (layout, flatten, unflatten_into, client_update, aggregate,
                            expected_steps, lr_at, ce_loss, make_optimizer, ACC_KEYS)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj.metrics import metrics_from_confusion, METRIC_KEYS

torch.manual_seed(0)
base = dict(MCFG, n_features=66, lr=1e-3, lr_schedule="cosine", lr_min=1e-5, rounds=50,
            weight_decay=1e-4, clip=1.0, batch=8, local_epochs=1, seed=42, device="cpu",
            eval_batch=16, num_classes=16, sparsity=0.7, prune_importance="group_l1")
n_pass = 0
def ok(cond, msg):
    global n_pass
    assert cond, msg; n_pass += 1; print("  ok", msg)

# ---- the unpruned DAGSNet
torch.manual_seed(42); F0 = build_full(base)
ok(n_params(F0) == N_PARAMS_FULL == 395_024, "unpruned DAGSNet has 395,024 params")
c0 = dict(base, sparsity=0.0, prune_plan=None)
torch.manual_seed(42); M00 = build_model(c0)
ok(all(torch.equal(a, b) for a, b in zip(M00.state_dict().values(), F0.state_dict().values())),
   "build_model without a plan is theta_0 itself")

# ---- the pruning plan (server initialization)
plan, info = compute_plan(base)
ok(plan is not None and info["n_params_full"] == 395_024, "compute_plan returns a plan + info")
ok(info["n_params"] == 35_891, f"sparsity 0.7 keeps 35,891 params (got {info['n_params']:,})")
ok(info["plan_id"] == plan_id(plan) and len(info["plan_id"]) == 16, "plan_id is the plan's hash")
plan2, info2 = compute_plan(base)
ok(plan2 == plan and info2["plan_id"] == info["plan_id"], "compute_plan is deterministic")
p0, i0 = compute_plan(dict(base, sparsity=0.0))
ok(p0 is None and i0["n_params"] == 395_024 and i0["plan_id"] is None, "sparsity 0 -> no plan")
# JSON round trip (the plan lives in cfg, config.json and every weights file)
plan_rt = json.loads(json.dumps(plan))
ok(plan_rt == plan and plan_id(plan_rt) == info["plan_id"], "plan survives a JSON round trip")
cfg = dict(base, prune_plan=plan_rt, plan_id=info["plan_id"], n_params=info["n_params"])
torch.manual_seed(42); S = build_model(cfg)
ok(n_params(S) == 35_891, "build_model(cfg with plan) has the pruned count")
# theta' = M ⊙ theta_0: every kept weight equals the corresponding weight of theta_0
sd0, sdp = F0.state_dict(), S.state_dict()
same = True
for name, sel in plan.items():
    w0, wp = sd0[name + ".weight"], sdp[name + ".weight"]
    o = torch.as_tensor(sel["out"])
    if sel["in"] is not None:
        same &= torch.equal(wp, w0[o][:, torch.as_tensor(sel["in"])])
    else:
        same &= torch.equal(wp, w0[o])
ok(same, "Eq. (8): every kept channel of theta' is the same weight in theta_0")
x = torch.randn(8, 66)
ok(S(x).shape == (8, 16), "pruned model: (B,66)->(B,16), 16 logits kept")
ok(all(len(plan[n]["out"]) == 16 for n in ("head.5",)), "final Linear keeps all 16 outputs")
ok(all(len(plan[n]["in"]) == 6 for n in ("stems.0.0", "stems.1.0", "stems.2.0", "stems.3.0")),
   "the 6 input channels (patch_len) are never pruned")
# uniform per-layer ratio: every conv / linear output keeps int(n * 0.3) channels (the
# LayerNorm and the head's input follow the CONCAT of pruned branches, 544 -> 157)
for n_, sel in plan.items():
    L = info["layers"][n_]
    if L["type"] in ("Conv1d", "Linear") and n_ != "head.5":
        assert L["out"][1] == int(L["out"][0] * 0.3), (n_, L)
ok(True, "every conv/linear keeps int(n * (1 - sparsity)) output channels (uniform local ratio)")
ok(info["layers"]["head.0"]["out"] == [544, 157] and info["layers"]["head.2"]["in"] == [544, 157],
   "the concat of the four pruned branches is 157 channels (was 544)")
# the kept channels are the L1-largest within a group: check the stem groups, whose
# importance is the L1 norm of the stem conv row + its BN + the consumers' in-columns
kept = plan["stems.0.0"]["out"]; dropped = sorted(set(range(96)) - set(kept))
ok(len(kept) == 28 and len(dropped) == 68, "stem 0: 28 of 96 channels kept")
# the plan must refuse a model it was not made for
try:
    apply_plan(build_full(dict(base, stem_ch=64)), plan); raise AssertionError
except RuntimeError:
    ok(True, "apply_plan refuses a model whose dimensions do not match the plan")
bad = copy.deepcopy(plan); bad["stems.0.0"]["out"][0] = 999
try:
    apply_plan(build_full(base), bad); raise AssertionError
except RuntimeError:
    ok(True, "apply_plan refuses an index outside the module")
bad = copy.deepcopy(plan); del bad["head.2"]
try:
    apply_plan(build_full(base), bad); raise AssertionError
except RuntimeError:
    ok(True, "apply_plan refuses a plan that skips a module")
# the pruned model keeps the cbr structure (Conv -> BN -> ReLU), so fold_bn applies
n_bn = sum(isinstance(m, nn.BatchNorm1d) for m in S.modules())
ok(n_bn == 31, f"pruned model still has 31 BatchNorm layers ({n_bn})")

# flat round trip on the pruned model
fk, ik, nP = layout(S)
ok(nP == 35_891, "layout puts parameters first")
fv, iv = flatten(S, fk, ik)
S2 = build_model(cfg); unflatten_into(S2, fv, iv, fk, ik)
S.eval(); S2.eval()
ok(torch.equal(S(x), S2(x)), "flatten/unflatten round trip: max|dlogit| = 0")

# BN folding exact (after moving BN stats off their init)
S.train()
for _ in range(3): S(torch.randn(32, 66) * 3 + 1)
S.eval()
Tf = fold_bn(build_model(cfg))
load_folded(Tf, S)
with torch.no_grad():
    d = (Tf(x) - S(x)).abs().max().item()
ok(d < 1e-4, f"fold_bn exact on the pruned model: max|dlogit| = {d:.2e}")
ok(all(not p.requires_grad for p in Tf.parameters()), "folded template has no grads")

# eval_model = full confusion over all rows including the tail batch
TX = torch.randn(37, 66).half(); TY = torch.randint(0, 16, (37,)).to(torch.uint8)
cm, nf, preds = eval_model(Tf, Tf, TX, TY, cfg, want_preds=True)
with torch.no_grad(): ref = Tf(TX.float()).argmax(1)
ok(cm.sum().item() == 37 and nf == 0, "eval covers every row, no tail dropped")
ok(torch.equal(preds.long(), ref), "eval preds == argmax of eager logits")
ok(np.array_equal(np.bincount(TY.long().numpy() * 16 + ref.numpy(), minlength=256).reshape(16, 16),
                  cm.numpy()), "confusion == bincount(y_true*C + y_pred)")
# sharded eval sums to the full matrix
cma, _, pa = eval_model(Tf, Tf, TX[:20], TY[:20], cfg, want_preds=True)
cmb, _, pb = eval_model(Tf, Tf, TX[20:], TY[20:], cfg, want_preds=True)
ok(torch.equal(cma + cmb, cm) and torch.equal(torch.cat([pa, pb]), preds),
   "two row shards sum to the full confusion matrix and concatenate to the same preds")

# ---- the objective
torch.manual_seed(3)
z = torch.randn(8, 16, requires_grad=True); y = torch.randint(0, 16, (8,))
loss = ce_loss(z, y)
p = F.softmax(z, 1)
ok(torch.allclose(loss, -(torch.log(p[torch.arange(8), y])).mean()),
   "Eq. (5)/(6): CE = -mean log softmax(z)[y]")

# ---- client_update: steps, the model moves, modes are right, equals plain AdamW training
X = torch.randn(20, 66).half(); Y = torch.randint(0, 16, (20,)).to(torch.uint8)
torch.manual_seed(1); Se = build_model(cfg)
s_before = flatten(Se, fk, ik)[0].clone()
opt = make_optimizer(Se, cfg["lr"], cfg, fused=False)
ok(sum(len(g["params"]) for g in opt.param_groups) == len(list(Se.parameters())),
   "the optimizer holds the model's parameters and only those")
sc = torch.amp.GradScaler("cuda", enabled=False)
gen = torch.Generator(); gen.manual_seed(7)
acc, n = client_update(Se, Se, opt, sc, X, Y, 0, 20, cfg, gen)
ok(n == expected_steps(20, cfg) == 3, "steps = ceil(20/8) = 3 (tail batch kept)")
ok(float(acc[len(ACC_KEYS)]) == 0 and float(acc[len(ACC_KEYS) + 1]) == 0,
   "no skips, no non-finite without AMP")
ok(all(math.isfinite(float(v)) for v in acc), "finite accumulators")
ok(not torch.equal(s_before[:nP], flatten(Se, fk, ik)[0][:nP]), "the weights moved")
ok(Se.training and all(p.requires_grad for p in Se.parameters()), "model left in train(), trainable")
acc2, n2 = client_update(Se, Se, opt, sc, X, Y, 0, 20, dict(cfg, local_epochs=2), gen)
ok(n2 == 6 == expected_steps(20, dict(cfg, local_epochs=2)), "local_epochs=2 -> 6 steps")
# the update is exactly local CE training with AdamW + clip (no hidden term)
torch.manual_seed(1); Sa = build_model(cfg)
torch.manual_seed(1); Sb = build_model(cfg)
oa = make_optimizer(Sa, cfg["lr"], cfg, fused=False)
ob = torch.optim.AdamW(Sb.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
g1 = torch.Generator(); g1.manual_seed(7); torch.manual_seed(11)
client_update(Sa, Sa, oa, sc, X, Y, 0, 20, cfg, g1)
g2 = torch.Generator(); g2.manual_seed(7); torch.manual_seed(11)
Sb.train()
perm_b = torch.randperm(20, generator=g2)      # ONE permutation per epoch, as the loop does
for i in range(0, 20, 8):
    idx = perm_b[i:i + 8]
    l = F.cross_entropy(Sb(X[idx].float()), Y[idx].long())
    l.backward(); torch.nn.utils.clip_grad_norm_(Sb.parameters(), 1.0)
    ob.step(); ob.zero_grad(set_to_none=True)
ok(torch.allclose(flatten(Sa, fk, ik)[0], flatten(Sb, fk, ik)[0], atol=1e-6),
   "client_update is exactly local CE training with AdamW + clip (Algorithm 5)")

# ---- aggregation: Eq. (9), unweighted mean over ALL N clients, int buffers take the max
ups = [(0, torch.ones(5), torch.tensor([1])), (1, torch.zeros(5), torch.tensor([4])),
       (2, torch.full((5,), 2.0), torch.tensor([2]))]
acc_, ints = aggregate(ups, 3)
ok(torch.allclose(acc_, torch.full((5,), 1.0)) and ints.item() == 4,
   "aggregate: (1 + 0 + 2) / 3 = 1 regardless of n_k; int buffers take the max")
for label, bad_ups, n_ in (("a missing client", ups[:2], 3), ("unsorted clients", ups[::-1], 3),
                           ("wrong N", ups, 4)):
    try: aggregate(bad_ups, n_); raise AssertionError(label)
    except ValueError: pass
ok(True, "aggregate refuses a missing client, an unsorted list and a wrong N")

# ---- per-round LR schedule: endpoints, monotone, symmetric midpoint, constant, out of range
lrs = [lr_at(cfg, r) for r in range(1, 51)]
ok(lrs[0] == 1e-3 and abs(lrs[-1] - 1e-5) < 1e-18, "cosine: lr(1) = lr, lr(T) = lr_min")
ok(all(a_ > b_ for a_, b_ in zip(lrs, lrs[1:])), "cosine: strictly decreasing over 50 rounds")
mid = lr_at(dict(cfg, rounds=51), 26)
ok(abs(mid - (1e-3 + 1e-5) / 2) < 1e-18, "cosine: midpoint of an odd horizon is the mean of the ends")
ok(abs(lr_at(cfg, 10) - (1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 9 / 49)))) < 1e-18,
   "cosine: round 10 matches the shared formula")
ok(lr_at(dict(cfg, lr_schedule="constant"), 37) == 1e-3, "constant: cfg['lr'] every round")
ok(lr_at(dict(cfg, rounds=1), 1) == 1e-3, "cosine with a 1-round horizon is the peak")
for bad in (0, 51):
    try: lr_at(cfg, bad); raise AssertionError(bad)
    except ValueError: pass
ok(True, "cosine: round outside 1..T raises")
try: lr_at(dict(cfg, lr_schedule="linear"), 1); raise AssertionError
except ValueError: ok(True, "unknown schedule raises")

# ---- metrics vs sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
yt = np.random.randint(0, 16, 5000); yp = np.random.randint(0, 16, 5000); yp[:2000] = yt[:2000]
cmx = np.bincount(yt * 16 + yp, minlength=256).reshape(16, 16)
m = metrics_from_confusion(cmx)
ref = {"accuracy": accuracy_score(yt, yp)}
for avg in ("macro", "micro", "weighted"):
    ref[f"precision_{avg}"] = precision_score(yt, yp, average=avg, zero_division=0)
    ref[f"recall_{avg}"] = recall_score(yt, yp, average=avg, zero_division=0)
    ref[f"f1_{avg}"] = f1_score(yt, yp, average=avg, zero_division=0)
dm = max(abs(m[k] - ref[k]) for k in METRIC_KEYS)
ok(dm < 1e-12, f"10 metrics vs sklearn: max|d| = {dm:.1e}")
print(f"\n{n_pass} checks passed")
