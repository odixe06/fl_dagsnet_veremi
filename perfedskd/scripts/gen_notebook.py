#!/usr/bin/env python
"""Generate one PerFed-SKD training notebook per client scenario.

The notebook body is assembled from papers/perfedskd-singh-2025/proj/*.py, which are the
same files the local tests exercise. Editing a module and regenerating is the only
supported path; never hand-edit the .ipynb.

    python scripts/gen_notebook.py --owner catbaochau                   # 20c 50c 100c
    python scripts/gen_notebook.py --owner catbaochau --clients 20 --probe --max-hours 2
    python scripts/gen_notebook.py --owner X --clients 100 --session 2 --require-resume \\
        --kernel-source X/perfedskd-veremi-100-clients --max-hours 11
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "papers/perfedskd-singh-2025/proj"
RUNTIME = json.loads((ROOT / "knowledge/runtime.json").read_text())
META = json.loads((ROOT / "knowledge/meta.json").read_text())
MODULES = ("model", "ckpt", "metrics", "data", "perfedskd", "evaluate", "driver", "verify")

# batch from knowledge/dataset.md §4 and the owner's instruction of 2026-09-21. Every
# client trains in every round (Algorithm 2 line 2 runs over all M in parallel), so the
# mean over clients is always a mean over models trained this round.
SCENARIOS = {20: dict(batch=512, dataset="veremi-fl-20client"),
             50: dict(batch=512, dataset="veremi-fl-50client"),
             100: dict(batch=256, dataset="veremi-fl-100client")}
# Owner's decisions of 2026-09-21 (CONTEXT.md §1): AdamW / wd 1e-4 as in
# knowledge/architecture.md, 50 rounds x 1 local epoch, per-round cosine LR 1e-3 -> 1e-5
# (proj/perfedskd.py::lr_at). The paper names no optimizer, no rate, no lambda and no
# divergence family; Eq. (3) says only "Stochastic Gradient Descent".
PAPER = dict(lr=1e-3, lr_schedule="cosine", lr_min=1e-5, weight_decay=1e-4,
             rounds=50, local_epochs=1, lam=1.0, select_metric="accuracy")


def kaggle_slug(title):
    """Kaggle's own rule: lowercase, every run of non-alphanumerics becomes one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "source": s,
                     "execution_count": None, "outputs": []}


CALIBRATION = '''# ---- PROBE ONLY: measure what sm_86 cannot tell us about the T4 before the rounds run.
# Train: one client's SKD epoch (frozen-teacher forward + student forward/backward per
# step) eager vs compiled. Eval: one folded model over the FULL test set, eager vs
# compiled, at two batch sizes -- eval is 45-65 % of a round here, so its batch is the
# single most valuable number this cell produces. Everything is freed afterwards so the
# two workers start with the whole GPU.
import gc, time, torch
from proj.model import build_model
from proj.perfedskd import (layout, flatten, unflatten_into, client_update,
                            make_optimizer, ACC_KEYS)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj import driver as D

CAL = {}
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cudnn.benchmark = True
for _n in ('recompile_limit', 'cache_size_limit'):
    if hasattr(torch._dynamo.config, _n): setattr(torch._dynamo.config, _n, 64)
cache = Path(CFG["cache"])
TX = D._resident(cache / "test_X.f16.npy", dev); TY = D._resident(cache / "test_y.u8.npy", dev)
lo, hi = spans[min(spans, key=lambda c: spans[c][1] - spans[c][0])]   # smallest client
hi = min(hi, lo + 400 * CFG["batch"])                                  # ~400 steps
X = torch.from_numpy(np.load(cache / "train_X.f16.npy", mmap_mode="r")[lo:hi].copy()).to(dev)
Y = torch.from_numpy(np.load(cache / "train_y.u8.npy", mmap_mode="r")[lo:hi].copy()).to(dev)

def one_client(Sc, Se, Vc, Ve):
    opt = make_optimizer(Se, CFG["lr"], CFG, fused=True)
    sc = torch.amp.GradScaler("cuda"); sc.scale(torch.zeros(1, device=dev))
    g = torch.Generator(device=dev); g.manual_seed(1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    acc, n = client_update(Sc, Se, Vc, Ve, opt, sc, X, Y, 0, hi - lo, CFG, g)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000, n, int(acc[len(ACC_KEYS)].item())

torch.manual_seed(CFG["seed"])
Se = build_model(CFG).to(dev).train(); Ve = build_model(CFG).to(dev).eval()
for _p in Ve.parameters(): _p.requires_grad_(False)
fk, ik, _ = layout(Se)
s0, i0 = flatten(Se, fk, ik); v0, vi0 = flatten(Ve, fk, ik)
def reset():
    unflatten_into(Se, s0, i0, fk, ik); unflatten_into(Ve, v0, vi0, fk, ik)
ms, n, sk = one_client(Se, Se, Ve, Ve)
CAL["train_eager_ms_per_step"] = ms
print(f"train eager   : {ms:.2f} ms per step ({n} steps, {sk} skipped) at batch {CFG['batch']}")
reset()
t0 = time.perf_counter()
Sc, Vc = D._compile_train(Se, Ve, CFG, dev, X[:CFG["batch"]].float(), Y[:CFG["batch"]].long())
CAL["compile_seconds"] = time.perf_counter() - t0
CAL["train_backend"] = "compiled" if Sc is not Se else "eager"
if Sc is not Se:
    reset(); one_client(Sc, Se, Vc, Ve)                          # warm the graphs
    reset(); ms, n, sk = one_client(Sc, Se, Vc, Ve)
    CAL["train_compiled_ms_per_step"] = ms
    print(f"train compiled: {ms:.2f} ms per step ({n} steps, {sk} skipped) | "
          f"{CAL['train_eager_ms_per_step']/ms:.2f}x | compile+gate {CAL['compile_seconds']:.0f}s")
del Sc, Vc

Te = fold_bn(build_model(CFG).to(dev)); load_folded(Te, Se)
for eb in (16384, 32768):
    c = dict(CFG, eval_batch=eb)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cm, nf, _ = eval_model(Te, Te, TX, TY, c); torch.cuda.synchronize()
    r = TX.shape[0] / (time.perf_counter() - t0); CAL[f"eval_eager_folded_{eb}"] = r
    print(f"eval eager-folded  batch {eb:>5}: {r:,.0f} rows/s")
    if CFG["compile"]:
        Tc = D._compile_eval(Te, c, dev, TX[:eb].float())
        if Tc is not Te:
            eval_model(Tc, Te, TX[:4 * eb], TY[:4 * eb], c)          # warm
            torch.cuda.synchronize(); t0 = time.perf_counter()
            cm2, nf2, _ = eval_model(Tc, Te, TX, TY, c); torch.cuda.synchronize()
            r = TX.shape[0] / (time.perf_counter() - t0); CAL[f"eval_compiled_folded_{eb}"] = r
            print(f"eval compiled-folded batch {eb:>5}: {r:,.0f} rows/s | "
                  f"|dCM|={int((cm2 - cm).abs().sum())} cells of {TX.shape[0]}")
            torch._dynamo.reset()
        del Tc
# What a full round would cost from these two numbers alone, before any round has run.
_mt = CAL.get("train_compiled_ms_per_step", CAL["train_eager_ms_per_step"]) / 1000.0
_re = max(CAL.get("eval_compiled_folded_16384", 0), CAL.get("eval_eager_folded_16384", 1))
_steps = sum(-(-(h - l) // CFG["batch"]) for l, h in spans.values()) * CFG["local_epochs"]
CAL["projected_train_sec"] = _steps * _mt / CFG["world_size"]
CAL["projected_eval_sec"] = (CFG["n_clients"] + 1) / CFG["world_size"] * TX.shape[0] / _re
CAL["projected_round_sec"] = CAL["projected_train_sec"] + CAL["projected_eval_sec"]
print(f"projected round: train {CAL['projected_train_sec']:.0f}s "
      f"({_steps:,} steps / {CFG['world_size']} GPUs) + eval {CAL['projected_eval_sec']:.0f}s "
      f"({CFG['n_clients']}+1 models) = {CAL['projected_round_sec']/60:.1f} min; "
      f"{CFG['rounds']} rounds = {CAL['projected_round_sec']*CFG['rounds']/3600:.1f} h")
del Te, Se, Ve, X, Y, TX, TY, s0, v0
gc.collect(); torch.cuda.empty_cache(); torch._dynamo.reset()
CAL["vram_after_free_gb"] = torch.cuda.memory_allocated(dev) / 2**30
print("calibration:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in CAL.items()}))
(C.run_dir(CFG["run_name"]) / "reports" / "calibration.json").write_text(json.dumps(CAL, indent=1))
if run is not None:
    run.summary.update({f"cal_{k}": v for k, v in CAL.items()})'''


def build(K, owner, wandb_project, require_resume=False, kernel_sources=(),
          probe=False, max_hours=11.0, run_tag="", dataset_sources=(), eval_batch=16384,
          session=0):
    sc = SCENARIOS[K]
    # A probe measures the T4 and is thrown away: its own run_name keeps its checkpoints,
    # its W&B run and its fingerprint separate from the real run of the same scenario.
    # run_tag does the same for a relaunch: W&B is opened with resume="allow" on a fixed
    # id, and run_name is in ckpt.FINGERPRINT_KEYS.
    run_name = f"perfedskd_{K}c_probe" if probe else f"perfedskd_{K}c{run_tag}"
    rounds = 2 if probe else PAPER["rounds"]
    # Kaggle derives the kernel slug from the TITLE and ignores `id` when the two disagree.
    # A continuation session gets its own slug (" s2" -> "-s2") and attaches the previous
    # session's kernel; run_name is unchanged so the fingerprint and the import still match.
    title = f"PerFed-SKD VeReMi {K} clients" + (" probe" if probe else "") \
        + (f" {run_tag.strip('_').replace('_', ' ')}" if run_tag else "") \
        + (f" s{session}" if session else "")
    slug = kaggle_slug(title)
    cells = [md(f"""# PerFed-SKD on VeReMi NextGen — {K} edge devices

Singh, Rupchandani & Adhikari, *Personalized Federated Learning for Heterogeneous Edge
Device: Self-Knowledge Distillation Approach* — Eq. (2)–(3), Algorithm 1 (server) and
Algorithm 2 (device), with **DAGSNet** (395,024 params) as every model in the system.
The paper's server-side teacher trained on a "predefined dataset" and any feature
extraction backbone are deliberately absent: the only knowledge transfer here is the one
the method is named after, from a device's own previous personalized model to its
current one.

| from the paper | from `knowledge/` and the owner's decisions |
|---|---|
| Eq. (2) φ_m(ω) = f_m(ω) + λ·L( x(V_m) ‖ x(ω) ); V_m = the device's model from the previous round, frozen | L = KL(p_teacher ‖ p_student) on softmax, temperature 1, λ = {PAPER['lam']}; teacher in `eval()` under `no_grad` |
| Alg. 2 l.4–7: selected devices take ω^t, the rest keep their own weights; **all** devices train | one local epoch, batch {sc['batch']} |
| Alg. 1 l.10: ω^{{t+1}} = (1/\\|S\\|) Σ_{{m∈S}} ω_m^{{t+1}} | only the selected devices upload |
| Alg. 1 l.11 + l.4–6: A ← (1/\\|M\\|) Σ a_m, τ = A, select m where a_m < τ | a_m = **accuracy** of device m's model on the full global test set |
| Eq. (3): "Stochastic Gradient Descent", no rate given | AdamW, wd {PAPER['weight_decay']}, re-created per device per round; lr {PAPER['lr']} → {PAPER['lr_min']} cosine per round; clip 1.0; fp16 AMP; seed 42 |

**Round 1** selects every device (no accuracies exist yet) and every V_m is ω^0, so the
distillation term starts at exactly 0 and the teacher becomes a genuinely different model
from round 2 on.

**Evaluation:** every round, every device's personalized model on the full
10,761,343-row test set — 10 metrics per device, mean/std/min/max over all {K} — plus the
server's aggregate ω^t as one extra model (`global_*` columns, never mixed into the
device mean). The same numbers feed the threshold, so the selection rule is measured on
exactly the artifacts that are published.

**Deviations, all deliberate** (full list in `docs/rebuild.md` §2): no server-side teacher and
no backbone; τ is Algorithm 1's mean of device accuracies, not the prose's "global model
accuracy" (which on this data would select every device in every round); all devices
train and only the selected ones upload, following §IV's prose over Algorithm 1's loop
bound; KL at temperature 1 with λ = {PAPER['lam']} where the paper gives no divergence and no λ;
AdamW with a per-round cosine schedule where the paper gives neither; the global test set
is shared by every device, so these scores measure global generalization of a
personalized model, not performance on the device's own distribution.

Per-round output is **weights only** (ω^t plus every ω_m^t as `state_dict` tensors, with
the selected subset and τ in the same file); `proj/ckpt.py::load_weights` rebuilds them at
any round, and the last cell re-derives every published metric — including τ and the
selection — from the confusion matrices on disk."""),

    code("""import os, subprocess, sys, time, torch
T0 = time.monotonic()     # session clock: the 12 h cap charges for spawn and compile too.
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}. machine_shape must be NvidiaTeslaT4."
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    assert cap == (7, 5), f"GPU {i} is {cap}, expected (7,5) Tesla T4"
    print(i, torch.cuda.get_device_name(i), cap,
          f"{torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GiB")
print("torch", torch.__version__, "| python", sys.version.split()[0])
# NCCL is unused (FL clients never form a process group) but P2P probing can still hang.
os.environ["NCCL_P2P_DISABLE"] = "1"; os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.makedirs("/kaggle/working/proj", exist_ok=True)
open("/kaggle/working/proj/__init__.py", "w").close()
sys.path.insert(0, "/kaggle/working")"""),

    code(f'''CFG = dict(
    # --- architecture: knowledge/architecture.md, frozen (395,024 params, every model)
    patch_len=6, stem_ch=96, dense_growth=32, dense_layers=3,
    incep_modules=2, fire_modules=3, dropout=0.1,
    num_classes=16, n_features=66,
    # --- PerFed-SKD: owner's decisions 2026-09-21 (the paper gives no optimizer, no rate,
    #     no lambda and no divergence family); LR cosine per round lr -> lr_min
    #     (proj/perfedskd.py::lr_at). `select_metric` only DECLARES the rule so the
    #     fingerprint can see it -- proj/perfedskd.py::SELECT_METRIC is the rule, and the
    #     driver refuses to start if the two disagree.
    lam={PAPER["lam"]}, select_metric="{PAPER["select_metric"]}",
    lr={PAPER["lr"]}, lr_schedule="{PAPER["lr_schedule"]}", lr_min={PAPER["lr_min"]},
    weight_decay={PAPER["weight_decay"]}, local_epochs={PAPER["local_epochs"]},
    rounds={rounds}, clip=1.0, seed=42,
    # --- compute: knowledge/dataset.md §4
    n_clients={K}, batch={sc["batch"]}, eval_batch={eval_batch},
    device="cuda", world_size=2, compile=True,
    # Each client starts a fresh GradScaler at 2**16 and spends a few steps calibrating.
    # Fixed before the first measurement so it cannot be widened afterwards.
    max_skips_per_client=16,
    preds_rounds={[] if probe else [rounds]},   # (N, 10.76 M) uint8 per listed round
    finalize_reserve_seconds=900,   # never start a round that leaves no time to commit it
    run_name="{run_name}",
    cache="/kaggle/temp/veremi_cache",
    max_seconds={max_hours} * 3600,   # 12 h hard cap; leave room to finalize artifacts
    require_resume={require_resume},
)
# data_id is filled in below, once the feature order and scaler are known. It is part of
# proj/ckpt.py FINGERPRINT_KEYS, so a run cannot resume across a changed preprocessing.
for k, v in CFG.items(): print(f"{{k:>20}} = {{v}}")''')]

    for mod in MODULES:
        body = (PROJ / f"{mod}.py").read_text().rstrip()
        cells.append(code(f"%%writefile /kaggle/working/proj/{mod}.py\n{body}"))

    cells += [code(f'''import wandb
from kaggle_secrets import UserSecretsClient

try:
    _wandb_key = UserSecretsClient().get_secret("wandb_key")
except Exception as e:
    raise SystemExit(f"W&B secret unavailable: {{e}}")
wandb.login(key=_wandb_key, relogin=True, verify=True)
del _wandb_key
# W&B fails fast before dataset decode or model preparation. The key is never persisted.
run = wandb.init(project="{wandb_project}", name="{run_name}", config=CFG,
                 resume="allow", id="{run_name}")
print("W&B:", run.url)''')]

    cells += [
    code(f'''# Cheap identity work, then the resume gate, then the decode. A continuation push must
# die at the gate, not after a two-minute parquet pass.
import hashlib, json, time, numpy as np
from pathlib import Path
from proj import ckpt as C
from proj.data import (find_root, load_clients, load_test, assert_fp16_safe,
                       cache_ok)

FL_ROOT = find_root("train/client_id=000")
CEN_ROOT = find_root("upload/test")
TEST_ROOT = CEN_ROOT / "upload/test"
SCALER = json.loads((CEN_ROOT / "upload/scaler.json").read_text())["features"]
assert len(SCALER) == 66, f"scaler has {{len(SCALER)}} entries, expected 66"
FEATS = {META["feature_cols"]!r}
CLASS_NAMES = {META["class_names"]!r}

# What the fingerprint could not otherwise see: a permuted feature order, a re-fitted
# scaler or a different partition keep every shape identical.
CFG["data_id"] = hashlib.sha256(json.dumps({{
    "features": FEATS, "classes": CLASS_NAMES, "n_clients": CFG["n_clients"],
    "scaler": [[SCALER[c]["mean"], SCALER[c]["std_used"]] for c in FEATS],
}}, sort_keys=True).encode()).hexdigest()[:16]
print("FL root    :", FL_ROOT)
print("test root  :", TEST_ROOT)
print("data_id    :", CFG["data_id"])
print("fingerprint:", C.fingerprint(CFG))

last = C.resolve_resume(CFG["run_name"], CFG)
if CFG["require_resume"] and last is None:
    raise SystemExit("require_resume set but no VERIFIED checkpoint found — fix the "
                     "attachment. A marker without its artifacts does not count.")
print("resume from round", last)

cache = Path(CFG["cache"]); cache.mkdir(parents=True, exist_ok=True)
MF = cache / "manifest.json"
FILES = ("train_X.f16.npy", "train_y.u8.npy", "test_X.f16.npy", "test_y.u8.npy",
         "spans.json")
want = {{"data_id": CFG["data_id"], "n_clients": CFG["n_clients"],
        "fl_root": str(FL_ROOT), "test_root": str(TEST_ROOT)}}

t0 = time.time()
if cache_ok(cache, want, CFG["n_clients"]):
    print("prepack cache reusable (manifest matches and every file checks out)")
else:
    for f in FILES: (cache / f).unlink(missing_ok=True)
    MF.unlink(missing_ok=True)
    X, Y, spans = load_clients(FL_ROOT, FEATS, CFG["n_clients"])
    print(f"train {{X.shape}} max|x|={{assert_fp16_safe(X,'train'):.1f}}")
    np.save(cache / "train_X.f16.npy", X); np.save(cache / "train_y.u8.npy", Y)
    json.dump({{str(k): v for k, v in spans.items()}}, open(cache / "spans.json", "w"))
    del X, Y
    TX, TY = load_test(TEST_ROOT, FEATS, SCALER)
    print(f"test  {{TX.shape}} max|x|={{assert_fp16_safe(TX,'test'):.1f}}")
    np.save(cache / "test_X.f16.npy", TX); np.save(cache / "test_y.u8.npy", TY)
    del TX, TY
    MF.write_text(json.dumps(want))                       # cache marker: absolutely last
    assert cache_ok(cache, want, CFG["n_clients"]), "the cache just written does not validate"

spans = {{int(k): tuple(v) for k, v in json.load(open(cache / "spans.json")).items()}}
CFG["n_test"] = len(np.load(cache / "test_y.u8.npy", mmap_mode="r"))
n_train = sum(h - l for l, h in spans.values())
assert n_train == 43_045_415, f"train rows {{n_train}} != 43,045,415"
assert CFG["n_test"] == 10_761_343, f"test rows {{CFG['n_test']}}"
assert len(spans) == CFG["n_clients"], f"{{len(spans)}} spans for {{CFG['n_clients']}} clients"

# content_id reads the labels that are actually cached, hit or miss: the row counts and
# the class histogram change when the partition or the file contents change.
_ytr = np.load(cache / "train_y.u8.npy", mmap_mode="r")
_yte = np.load(cache / "test_y.u8.npy", mmap_mode="r")
assert len(_ytr) == n_train, f"train X/y disagree: {{len(_ytr)}} labels for {{n_train}} rows"
CFG["content_id"] = hashlib.sha256(json.dumps({{
    "clients": [[c, spans[c][0], spans[c][1]] for c in sorted(spans)],
    "train_hist": np.bincount(np.asarray(_ytr), minlength=16).tolist(),
    "test_hist": np.bincount(np.asarray(_yte), minlength=16).tolist(),
}}, sort_keys=True).encode()).hexdigest()[:16]
del _ytr, _yte
print("content_id :", CFG["content_id"])
print(f"prepack {{time.time()-t0:.1f}}s | {{n_train:,}} train / {{CFG['n_test']:,}} test rows")''')]

    if probe:
        cells.append(code(CALIBRATION))

    cells += [
    code('''from proj.driver import run as train, write_manifest
# Raises if this run's checkpoints were trained on different data. The resume gate above
# ran before the decode and could only compare data_id; content_id is the post-decode one.
write_manifest(CFG, CLASS_NAMES, spans,
               y_true_src=Path(CFG["cache"]) / "test_y.u8.npy",
               extra={"fl_root": str(FL_ROOT), "test_root": str(TEST_ROOT),
                      "feature_cols": FEATS, "n_test": CFG["n_test"],
                      "data_id": CFG["data_id"], "content_id": CFG["content_id"],
                      "scaler": {c: [SCALER[c]["mean"], SCALER[c]["std_used"]]
                                 for c in FEATS}})
hist = train(CFG, spans, CLASS_NAMES, wandb_run=run, t_origin=T0)
print(f"\\ncompleted {len(hist)} rounds this session")'''),

    code('''# Every published number, re-derived from the artifacts on disk. Never from memory.
import csv
from proj.verify import verify_run
from proj.model import build_model, N_PARAMS
from proj.metrics import METRIC_KEYS

d = C.run_dir(CFG["run_name"])
# y_true from the RUN, not from /kaggle/temp: the cache is gone with the session, and the
# check has to be the same one someone can repeat after downloading the output alone.
ok, lines = verify_run(d, cfg=CFG, build_model=build_model, expect_params=N_PARAMS,
                       y_true_path=d / "reports" / "y_true.u8.npy", full=True)
print("\\n".join(lines))

last = C.last_complete_round(d, C.fingerprint(CFG)) or 0
print(f"\\nrounds verified : {last} / {CFG['rounds']}")
if last < CFG["rounds"]:
    print(f"  INCOMPLETE — attach this notebook's output (or a checkpoint dataset of it) to "
          f"the next push and regenerate with --require-resume to continue at round {last + 1}")
rows = [r for r in csv.DictReader(open(d / "history.csv")) if int(r["round"]) <= last]
if rows:
    fin = rows[-1]
    # The headline is the LAST round, fixed before the run. best-f1 is chosen on the test
    # set after seeing it, so it is a description of the curve and not a second result.
    print(f"\\nresult at round {fin['round']} (mean over {CFG['n_clients']} personalized models; "
          f"std / min / max of f1_macro {float(fin['f1_macro_std']):.4f} / "
          f"{float(fin['f1_macro_min']):.4f} / {float(fin['f1_macro_max']):.4f}):")
    for k in METRIC_KEYS: print(f"  {k:<20} {float(fin[k]):.6f}   server aggregate {float(fin['global_' + k]):.6f}")
    sel = [int(r["n_selected"]) for r in rows]
    print(f"\\nselection: |S| per round min {min(sel)} median {sorted(sel)[len(sel)//2]} "
          f"max {max(sel)} of {CFG['n_clients']}; mean downlink+uplink saving "
          f"{100*sum(float(r['comm_saving']) for r in rows)/len(rows):.1f}% against "
          f"broadcasting to every device")
    b = max(rows, key=lambda r: float(r["f1_macro"]))
    print(f"\\n[descriptive only] best mean f1_macro {float(b['f1_macro']):.6f} "
          f"at round {b['round']} — picked on test, not a reported result")

# Calibration, from THIS session's rounds only (the CSV would mix in imported rounds).
sec = [float(r["seconds"]) for r in hist]
overhead = (time.monotonic() - T0) - sum(sec)
print(f"\\nbackend  : train {CFG.get('backend', '?')} | eval {CFG.get('backend_eval', '?')}")
print(f"session  : {len(hist)} round(s) here | startup+prepack+compile {overhead/60:.1f} min"
      f" | verify and W&B are outside this figure")
if len(sec) < 2:
    print("timing   : need 2 completed rounds to separate startup from steady state; "
          f"got {len(sec)}. No projection.")
else:
    steady = sum(sec[1:]) / len(sec[1:])
    vt = max(float(r.get("vram_train_gb", 0) or 0) for r in hist)
    ve = max(float(r.get("vram_eval_gb", 0) or 0) for r in hist)
    tr = sum(float(r["train_sec"]) for r in hist[1:]) / len(hist[1:])
    ev = sum(float(r["eval_sec"]) for r in hist[1:]) / len(hist[1:])
    print(f"timing   : rounds {[round(x) for x in sec[-3:]]}s | steady {steady:.0f}s/round "
          f"= train {tr:.0f}s + eval {ev:.0f}s ({hist[-1]['evaluated']} clients + aggregate) + commit")
    print(f"VRAM     : train {vt:.2f} GiB/GPU | eval {ve:.2f} GiB/GPU (of 16)")
    print(f"projected: {CFG['rounds']} rounds = "
          f"{(steady*CFG['rounds'] + overhead)/3600:.2f} h "
          f"({(steady*CFG['rounds'])/3600:.2f} h of rounds + {overhead/3600:.2f} h startup)")
if run is not None: run.finish()
assert ok, "artifact verification FAILED — see the FAIL lines above"''')]

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                   "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"}}}
    for c in nb["cells"]:
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
    meta = {"id": f"{owner}/{slug}", "title": title,
            "code_file": f"{run_name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "machine_shape": RUNTIME["machine_shape"],
            "docker_image": RUNTIME["docker_image"],
            # The two data mounts first, then any checkpoint dataset. A previous session's
            # run tree re-uploaded as a dataset OWNED BY THE ACCOUNT THAT RUNS is the only
            # resume route across accounts (kernel_sources silently mounts nothing then).
            "dataset_sources": [f"odixe0502/{sc['dataset']}",
                                "odixe0502/veremi-nextgen2026-centralized",
                                *dataset_sources],
            "kernel_sources": list(kernel_sources), "competition_sources": []}
    return nb, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--wandb-project", default="perfedskd-veremi")
    ap.add_argument("--clients", type=int, nargs="*", default=[20, 50, 100])
    ap.add_argument("--run-tag", default="",
                    help="suffix for run_name: new W&B run id, new checkpoint dir, new "
                         "fingerprint. Use it for a relaunch of a scenario that has already "
                         "logged rounds under the untagged name.")
    ap.add_argument("--probe", action="store_true",
                    help="2-round calibration run with a T4 micro-benchmark cell: separate "
                         "run_name, slug and W&B id")
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="driver stops before a round would pass this many session hours")
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--session", type=int, default=0,
                    help="continuation session number N >= 2: own slug '-sN' and output "
                         "dir '<K>c_sN'; pair with --require-resume --kernel-source")
    ap.add_argument("--require-resume", action="store_true",
                    help="continuation push: die unless a verified checkpoint is attached")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="owner/slug of the previous run whose output carries the "
                         "checkpoint (same account only); repeatable")
    ap.add_argument("--dataset-source", action="append", default=[],
                    help="owner/slug of a checkpoint dataset holding the previous "
                         "session's run tree; the cross-account resume route; repeatable")
    a = ap.parse_args()
    if a.require_resume and not (a.kernel_source or a.dataset_source):
        ap.error("--require-resume without --kernel-source or --dataset-source would fail "
                 "at the gate every time")
    if a.session and (a.session < 2 or not a.require_resume or a.probe):
        ap.error("--session N needs N >= 2 and --require-resume, and is not for a probe")
    for K in a.clients:
        nb, meta = build(K, a.owner, a.wandb_project, a.require_resume, a.kernel_source,
                         a.probe, a.max_hours, a.run_tag, a.dataset_source, a.eval_batch,
                         a.session)
        name = meta["code_file"]
        out = ROOT / "papers/perfedskd-singh-2025/notebook" / (
            f"{K}c_probe" if a.probe else f"{K}c_s{a.session}" if a.session else f"{K}c")
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(json.dumps(nb, indent=1))
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{K:>4}c -> {out}/{name}  ({len(nb['cells'])} cells)")
