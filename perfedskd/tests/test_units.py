"""Fast CPU unit checks: parameter counts, flat round-trip, BN folding exactness, the
SKD loss against a hand-written Eq. (2) (values AND the fact that no gradient reaches the
teacher), the teacher staying frozen across a client update, aggregation over the selected
subset only, the threshold and selection rule of Algorithm 1, the LR schedule, and the 10
metrics against sklearn."""
import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/perfedskd-singh-2025"))
from proj.model import build_model, CFG as MCFG, N_PARAMS
from proj.perfedskd import (layout, flatten, unflatten_into, client_update, aggregate,
                            expected_steps, lr_at, skd_loss, make_optimizer, threshold,
                            select_clients, ACC_KEYS, SELECT_METRIC)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj.metrics import metrics_from_confusion, METRIC_KEYS

torch.manual_seed(0)
cfg = dict(MCFG, n_features=66, lr=1e-3, lr_schedule="cosine", lr_min=1e-5, rounds=50,
           weight_decay=1e-4, clip=1.0, batch=8, local_epochs=1, seed=42, device="cpu",
           eval_batch=16, num_classes=16, lam=1.0, select_metric="accuracy")
n_pass = 0
def ok(cond, msg):
    global n_pass
    assert cond, msg; n_pass += 1; print("  ok", msg)

S = build_model(cfg)
ok(sum(p.numel() for p in S.parameters()) == N_PARAMS, "DAGSNet has 395,024 params")
x = torch.randn(8, 66)
ok(S(x).shape == (8, 16), "(B,66)->(B,16)")

# flat round trip
fk, ik, nP = layout(S)
ok(nP == N_PARAMS, "layout puts parameters first")
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
ok(d < 1e-4, f"fold_bn exact: max|dlogit| = {d:.2e}")
ok(all(not p.requires_grad for p in Tf.parameters()), "folded template has no grads")

# eval_model = full confusion over all rows including the tail batch
TX = torch.randn(37, 66).half(); TY = torch.randint(0, 16, (37,)).to(torch.uint8)
cm, nf, preds = eval_model(Tf, Tf, TX, TY, cfg, want_preds=True)
with torch.no_grad(): ref = Tf(TX.float()).argmax(1)
ok(cm.sum().item() == 37 and nf == 0, "eval covers every row, no tail dropped")
ok(torch.equal(preds.long(), ref), "eval preds == argmax of eager logits")
ok(np.array_equal(np.bincount(TY.long().numpy() * 16 + ref.numpy(), minlength=256).reshape(16, 16),
                  cm.numpy()), "confusion == bincount(y_true*C + y_pred)")

# ---- skd_loss against a hand-written Eq. (2)
torch.manual_seed(3)
z = torch.randn(8, 16, requires_grad=True)
z_t = torch.randn(8, 16, requires_grad=True)        # deliberately requires grad: see below
y = torch.randint(0, 16, (8,))
loss, (phi, ce, kd) = skd_loss(z, z_t, y, cfg["lam"])
p_t = F.softmax(z_t, 1)
h_ce = F.cross_entropy(z, y)
h_kd = (p_t * (p_t.log() - F.log_softmax(z, 1))).sum(1).mean()      # KL(p_t || p_student)
ok(torch.allclose(ce, h_ce), "Eq. (2): f_m is the cross-entropy of the student")
ok(torch.allclose(kd, h_kd, atol=1e-6), "L( x(V_m) || x(omega) ) = KL(p_teacher || p_student)")
ok(torch.allclose(phi, h_ce + cfg["lam"] * h_kd) and torch.allclose(loss, phi),
   "phi = CE + lam * KL, and the returned objective is phi itself")
# lambda scales only the distillation term
_, (phi2, ce2, kd2) = skd_loss(z, z_t, y, 2.0)
ok(torch.allclose(ce2, ce) and torch.allclose(kd2, kd)
   and torch.allclose(phi2, ce + 2.0 * kd), "lam scales the KL term and nothing else")
# a teacher equal to the student makes the term exactly 0 -- round 1 of the protocol
_, (_, _, kd0) = skd_loss(z, z.detach().clone(), y, cfg["lam"])
ok(abs(float(kd0.detach())) < 1e-7, "teacher == student => KL = 0 (round 1 starts with no distillation)")
# gradient structure: the teacher must receive NOTHING, and the student's gradient must be
# the gradient of CE + lam*KL with the teacher treated as a constant.
loss.backward()
ok(z_t.grad is None or float(z_t.grad.abs().max()) == 0.0,
   "no gradient reaches the teacher's logits (V_m is frozen)")
z2 = z.detach().clone().requires_grad_(True)
ref_s = F.cross_entropy(z2, y) + cfg["lam"] * (
    p_t.detach() * (p_t.detach().log() - F.log_softmax(z2, 1))).sum(1).mean()
ref_s.backward()
ok(torch.allclose(z.grad, z2.grad, atol=1e-6),
   "one backward of phi == grad of CE + lam*KL with a constant teacher")

# ---- client_update: steps, the student moves, the TEACHER DOES NOT, modes are right
X = torch.randn(20, 66).half(); Y = torch.randint(0, 16, (20,)).to(torch.uint8)
torch.manual_seed(1); Se, Ve = build_model(cfg), build_model(cfg)
Ve.eval()
for p in Ve.parameters(): p.requires_grad_(False)
s_before = flatten(Se, fk, ik)[0].clone(); v_before = flatten(Ve, fk, ik)[0].clone()
opt = make_optimizer(Se, cfg["lr"], cfg, fused=False)
ok(sum(len(g["params"]) for g in opt.param_groups) == len(list(Se.parameters())),
   "the optimizer holds the student's parameters and only those")
sc = torch.amp.GradScaler("cuda", enabled=False)
gen = torch.Generator(); gen.manual_seed(7)
acc, n = client_update(Se, Se, Ve, Ve, opt, sc, X, Y, 0, 20, cfg, gen)
ok(n == expected_steps(20, cfg) == 3, "steps = ceil(20/8) = 3 (tail batch kept)")
ok(float(acc[len(ACC_KEYS)]) == 0 and float(acc[len(ACC_KEYS) + 1]) == 0,
   "no skips, no non-finite without AMP")
ok(all(math.isfinite(float(v)) for v in acc), "finite accumulators")
s_after = flatten(Se, fk, ik)[0]; v_after = flatten(Ve, fk, ik)[0]
ok(not torch.equal(s_before[:N_PARAMS], s_after[:N_PARAMS]), "the student's weights moved")
ok(torch.equal(v_before, v_after),
   "the teacher is unchanged INCLUDING its BatchNorm buffers (eval() keeps running stats "
   "from being updated by the client's data)")
ok(Se.training and not Ve.training, "student left in train(), teacher left in eval()")
ok(all(p.requires_grad for p in Se.parameters()), "student left trainable")
ok(float(acc[0]) > float(acc[1]), "phi > ce: the distillation term is positive")
# local_epochs=2 doubles the step count
acc2, n2 = client_update(Se, Se, Ve, Ve, opt, sc, X, Y, 0, 20, dict(cfg, local_epochs=2), gen)
ok(n2 == 6 == expected_steps(20, dict(cfg, local_epochs=2)), "local_epochs=2 -> 6 steps")
# lam = 0 reduces the update to plain local training (the FedAvg-style ablation)
torch.manual_seed(1); Sa, Va = build_model(cfg), build_model(cfg)
Va.eval()
torch.manual_seed(1); Sb = build_model(cfg)
oa = make_optimizer(Sa, cfg["lr"], dict(cfg, lam=0.0), fused=False)
ob = torch.optim.AdamW(Sb.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
g1 = torch.Generator(); g1.manual_seed(7)
torch.manual_seed(11)
client_update(Sa, Sa, Va, Va, oa, sc, X, Y, 0, 20, dict(cfg, lam=0.0), g1)
g2 = torch.Generator(); g2.manual_seed(7)
torch.manual_seed(11)
Sb.train()
perm_b = torch.randperm(20, generator=g2)      # ONE permutation per epoch, as the loop does
for i in range(0, 20, 8):
    idx = perm_b[i:i + 8]
    l = F.cross_entropy(Sb(X[idx].float()), Y[idx].long())
    l.backward(); torch.nn.utils.clip_grad_norm_(Sb.parameters(), 1.0)
    ob.step(); ob.zero_grad(set_to_none=True)
ok(torch.allclose(flatten(Sa, fk, ik)[0], flatten(Sb, fk, ik)[0], atol=1e-6),
   "lam = 0 is exactly local CE training (the objective has no other hidden term)")

# ---- aggregation: unweighted mean over the SELECTED subset, int buffers take the max
ups = [(0, torch.ones(5), torch.tensor([1])), (3, torch.zeros(5), torch.tensor([4])),
       (7, torch.full((5,), 2.0), torch.tensor([2]))]
acc_, ints = aggregate(ups)
ok(torch.allclose(acc_, torch.full((5,), 1.0)) and ints.item() == 4,
   "aggregate: (1 + 0 + 2) / 3 = 1 regardless of n_k; int buffers take the max")
ok(torch.allclose(aggregate(ups[:2])[0], torch.full((5,), 0.5)),
   "aggregate divides by |S|, not by the number of clients in the system")
try:
    aggregate([]); raise AssertionError("empty S must not silently average")
except ValueError:
    ok(True, "aggregate([]) raises: an empty S is the driver's decision, not a 0/0")

# ---- Algorithm 1 lines 4-6 and 11: the threshold and the subset
a = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40}
tau = threshold(a)
ok(abs(tau - 0.25) < 1e-15, "tau = mean of the clients' accuracies (Algorithm 1 line 11)")
ok(select_clients(a, tau) == [0, 1], "devices strictly below tau are selected")
ok(select_clients({0: 0.5, 1: 0.5}, threshold({0: 0.5, 1: 0.5})) == [],
   "all-equal accuracies select nobody -- the driver must keep the previous global model")
ok(select_clients({0: 0.5, 1: 0.5, 2: 0.2}, 0.4) == [2], "an explicit tau is honoured")
ok(SELECT_METRIC == "accuracy", "the selection metric is the paper's accuracy")
ok(SELECT_METRIC in METRIC_KEYS, "the selection metric is one of the 10 stored metrics")
big = {c: float(v) for c, v in enumerate(np.random.rand(97))}
t2 = threshold(big)
sel = select_clients(big, t2)
ok(sel == sorted(sel) and all(big[c] < t2 for c in sel)
   and all(big[c] >= t2 for c in set(big) - set(sel)),
   "select_clients returns exactly {m : a_m < tau}, sorted")

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
