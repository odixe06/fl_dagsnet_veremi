"""Checkpoint contract, resume import, handoff bundle, merge, and the verifier's tamper
sensitivity.

Fixture: a tiny 3-client run (2 rounds) written with the real proj/ckpt.py writers and
real metrics from synthetic confusion matrices, on the PRUNED model. Every tamper case
below once passed a verifier in a sibling project; each must FAIL here.
"""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/lwfednids-bouayad-2024"))
from proj.model import build_model, CFG, n_params
from proj.prune import compute_plan
from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
from proj.lwfednids import expected_steps, lr_at, ACC_KEYS
from proj.verify import verify_run
from proj import ckpt as C

N, NT = 3, 1600
CFGD = {**CFG, "n_features": 66, "lr": 1e-3, "lr_schedule": "cosine", "lr_min": 1e-5,
        "weight_decay": 1e-4, "clip": 1.0, "sparsity": 0.7, "prune_importance": "group_l1",
        "n_clients": N, "batch": 64, "local_epochs": 1,
        "seed": 42, "data_id": "d" * 16, "run_name": "r", "rounds": 2,
        "eval_batch": 4096, "world_size": 1, "compile": False, "cache": "/tmp/x",
        "max_seconds": 60, "require_resume": False, "device": "cpu", "n_test": NT,
        "preds_rounds": [2]}
_plan, _info = compute_plan(CFGD)
CFGD.update(prune_plan=_plan, plan_id=_info["plan_id"], n_params=_info["n_params"])
CLASSES = [f"c{i}" for i in range(16)]
Y_TRUE = np.repeat(np.arange(16, dtype=np.uint8), NT // 16)


def mk(d):
    for s in C.SUBDIRS: (d / s).mkdir(parents=True, exist_ok=True)
    return d


def write_run(d, cfg, seed=0, rounds=2, mark_upto=None):
    """A run written with the production writers. Confusion matrices come from random
    predictions so the metrics are real numbers, not constants."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G = build_model(cfg)
    for r in range(1, rounds + 1):
        with torch.no_grad():
            for p in G.parameters(): p.add_(torch.randn_like(p) * 0.01)
        yp = np.where(rng.random(NT) < 0.6, Y_TRUE, rng.integers(0, 16, NT)).astype(np.uint8)
        cm = np.bincount(Y_TRUE.astype(int) * 16 + yp, minlength=256).reshape(16, 16)
        gm = metrics_from_confusion(cm)
        row = {"round": r, "lr": lr_at(cfg, r), "n_params": cfg["n_params"],
               "sparsity": cfg["sparsity"], **gm}
        clients = []
        for c in range(N):
            s = expected_steps(100, cfg)
            clients.append({"cid": c, "n_k": 100, "rank": 0, "seed": 1, "lr": lr_at(cfg, r),
                            "steps": s, "applied": s, "skipped": 0, "nonfinite": 0,
                            "sec": 1.0, "vram_gb": 0.0,
                            "loss": 0.5 + 0.1 * c + 0.01 * r, "gnorm": 1.0 + 0.2 * c})
        for k in ACC_KEYS:
            v = np.array([e[k] for e in clients])
            row[f"{k}_client_mean"], row[f"{k}_client_std"] = float(v.mean()), float(v.std())
            row[f"{k}_client_min"], row[f"{k}_client_max"] = float(v.min()), float(v.max())
        row.update({"sec_client_mean": 1.0, "steps": N * s, "applied": N * s, "skipped": 0,
                    "model_mib": 0.14, "comm_mib": 0.84, "train_sec": 1.0, "eval_sec": 1.0,
                    "vram_train_gb": 0.0, "vram_eval_gb": 0.0, "backend": "eager",
                    "seconds": 1.0})
        C.save_round_weights(C.cpu_sd(G), r, cfg, gm, d)
        C.atomic_np_save(d / "confusion" / f"round_{r:03d}.npy", cm)
        if r in cfg["preds_rounds"]:
            C.atomic_np_save(d / "preds" / f"round_{r:03d}.u8.npy", yp)
        (d / "logs" / f"round_{r:03d}.json").write_text(json.dumps({"round": r, "clients": clients}))
        (d / "metrics" / f"round_{r:03d}.json").write_text(json.dumps(
            {**row, "per_class": per_class_from_confusion(cm, CLASSES), "clients": clients}))
        C.append_history(d, row, [{"round": r, **c} for c in clients])
        if mark_upto is None or r <= mark_upto:
            C.mark_complete(d, r)
    (d / "reports").mkdir(exist_ok=True)
    (d / "reports" / "manifest.json").write_text(json.dumps(
        {"cfg": cfg, "fingerprint": C.fingerprint(cfg), "n_clients": N, "n_params": cfg["n_params"],
         "client_rows": {str(c): 100 for c in range(N)}}))
    with open(d / "reports" / "y_true.u8.npy", "wb") as f: np.save(f, Y_TRUE)
    return d


def check(d, cfg, require=2):
    return verify_run(d, cfg=cfg, build_model=build_model, expect_params=cfg["n_params"],
                      y_true_path=d / "reports" / "y_true.u8.npy", require_rounds=require)


def main():
    tmp = Path(tempfile.mkdtemp()); n = 0
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs" / name)
    cfg = dict(CFGD)
    d = write_run(mk(tmp / "runs/r"), cfg)
    fp = C.fingerprint(cfg)
    ok, lines = check(d, cfg); assert ok, "\n".join(lines)
    print("1. clean fixture passes the verifier"); n += 1

    # ---- weights contract
    raw = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    assert set(raw) == {"round", "global", "cfg", "fingerprint", "prev_sha", "metrics"}
    assert all(torch.is_tensor(v) for v in raw["global"].values())
    assert "running_mean" in " ".join(raw["global"]) and any("num_batches_tracked" in k for k in raw["global"])
    assert raw["cfg"]["prune_plan"] == cfg["prune_plan"] and raw["cfg"]["n_params"] == 35_891
    assert raw["global"]["stems.0.0.weight"].shape == (28, 6, 1)
    rs = torch.load(d / "resume/round_002.pt", map_location="cpu", weights_only=True)
    assert rs["round"] == 2 and "rng" in rs
    Gm, ck = C.load_weights(d / "weights/round_002.pt", build_model, cfg["n_params"])
    assert not Gm.training and n_params(Gm) == 35_891 and ck["round"] == 2
    G2, _ = C.load_weights(d / "weights/round_002.pt", build_model)        # count from the file's cfg
    x = torch.randn(4, 66)
    assert torch.equal(Gm(x), G2(x))
    # a file whose cfg declares another parameter count cannot be the model it holds
    bad = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    bad["cfg"] = dict(bad["cfg"], n_params=395_024)
    C.atomic_save(bad, tmp / "badn.pt")
    try:
        C.load_weights(tmp / "badn.pt", build_model); raise AssertionError
    except RuntimeError as e:
        assert "parameters" in str(e)
    # a file whose plan was swapped for another mask rebuilds a different shape: strict fails
    other, oinfo = compute_plan(dict(cfg, sparsity=0.5))
    bad = torch.load(d / "weights/round_002.pt", map_location="cpu", weights_only=True)
    bad["cfg"] = dict(bad["cfg"], prune_plan=other, plan_id=oinfo["plan_id"], sparsity=0.5,
                      n_params=oinfo["n_params"])
    bad["fingerprint"] = C.fingerprint(bad["cfg"])
    C.atomic_save(bad, tmp / "badplan.pt")
    try:
        C.load_weights(tmp / "badplan.pt", build_model); raise AssertionError
    except RuntimeError:
        pass
    print(f"2. weights: tensors only, BN buffers included, weights_only=True, plan inside cfg, "
          f"strict rebuild of the pruned model, wrong n_params / wrong plan refused, "
          f"{(d/'weights/round_002.pt').stat().st_size/1e6:.2f} MB"); n += 1

    # ---- fingerprint sees every scientific key, is blind to operational ones
    for k, v in (("lr", 0.01), ("local_epochs", 2), ("clip", 0.5),
                 ("weight_decay", 0.0), ("batch", 128), ("seed", 1), ("n_clients", 5),
                 ("data_id", "x" * 16), ("run_name", "other"), ("dropout", 0.2),
                 ("lr_schedule", "constant"), ("lr_min", 1e-6), ("rounds", 99),
                 # the mask is the architecture
                 ("sparsity", 0.5), ("plan_id", "0" * 16), ("prune_importance", "l2")):
        assert C.fingerprint({**cfg, k: v}) != fp, k
    for k, v in (("eval_batch", 1), ("max_seconds", 1), ("compile", True),
                 ("world_size", 2), ("preds_rounds", []), ("n_params", 1)):
        assert C.fingerprint({**cfg, k: v}) == fp, k
    try: C.fingerprint({k: v for k, v in cfg.items() if k != "clip"}); raise AssertionError
    except KeyError: pass
    print("3. fingerprint: 16 scientific keys move it, 6 operational keys do not, missing key raises"); n += 1

    # ---- tamper cases: each must FAIL
    def case(label, mutate):
        nonlocal n
        e = mk(tmp / "runs" / f"t_{n}")
        shutil.copytree(d, e, dirs_exist_ok=True)
        mutate(e)
        ok, lines = check(e, cfg)
        assert not ok, f"{label}: verifier passed a tampered run"
        print(f"4. tamper rejected: {label} -> {[l for l in lines if 'FAIL' in l][0].strip()[:90]}"); n += 1

    def _rj(e): return json.loads((e / "metrics/round_002.json").read_text())
    def _wj(e, j): (e / "metrics/round_002.json").write_text(json.dumps(j))
    def _rl(e): return json.loads((e / "logs/round_002.json").read_text())
    def _wl(e, j): (e / "logs/round_002.json").write_text(json.dumps(j))
    def m_nan(e):
        j = _rj(e); j["f1_macro"] = float("nan"); _wj(e, j)
    def m_json(e):
        j = _rj(e); j["f1_macro"] = 0.999; _wj(e, j)
    def m_w(e):
        w = torch.load(e / "weights/round_002.pt", map_location="cpu", weights_only=True)
        w["metrics"]["f1_macro"] = 0.999; C.atomic_save(w, e / "weights/round_002.pt")
    def m_cm(e):
        g = np.load(e / "confusion/round_002.npy"); g[0, 0] += 1; g[0, 1] -= 1
        C.atomic_np_save(e / "confusion/round_002.npy", g)
    def m_csv(e):
        rows = (e / "history.csv").read_text().splitlines(); (e / "history.csv").write_text("\n".join(rows[:2]) + "\n")
    def m_csv_val(e):
        rows = (e / "history.csv").read_text().splitlines()
        col = rows[0].split(",").index("accuracy"); f = rows[2].split(","); f[col] = "0.999"
        rows[2] = ",".join(f); (e / "history.csv").write_text("\n".join(rows) + "\n")
    def m_ccsv(e):
        rows = (e / "clients.csv").read_text().splitlines()
        col = rows[0].split(",").index("loss"); f = rows[-1].split(","); f[col] = "0.999"
        rows[-1] = ",".join(f); (e / "clients.csv").write_text("\n".join(rows) + "\n")
    def m_pc(e):
        j = _rj(e); j["per_class"][3]["f1"] = 999; _wj(e, j)
    def m_log_missing_client(e):
        lg = _rl(e); lg["clients"] = lg["clients"][:2]; _wl(e, lg)
        j = _rj(e); j["clients"] = j["clients"][:2]; _wj(e, j)
    def m_log_json_disagree(e):
        lg = _rl(e); lg["clients"][0]["loss"] = 7.0; _wl(e, lg)
    def m_steps(e):
        lg = _rl(e); lg["clients"][0]["applied"] = 0; lg["clients"][0]["skipped"] = lg["clients"][0]["steps"]; _wl(e, lg)
        j = _rj(e); j["clients"] = lg["clients"]; _wj(e, j)
    def m_nonfin(e):
        lg = _rl(e); lg["clients"][1]["nonfinite"] = 1; _wl(e, lg)
        j = _rj(e); j["clients"] = lg["clients"]; _wj(e, j)
    def m_client_mean(e):
        j = _rj(e); j["loss_client_mean"] = 0.999; _wj(e, j)
    def m_steps_total(e):
        j = _rj(e); j["steps"] = 1; _wj(e, j)
    def m_preds(e):
        p = np.load(e / "preds/round_002.u8.npy"); p[:50] = (p[:50] + 1) % 16
        C.atomic_np_save(e / "preds/round_002.u8.npy", p)
    def m_nopreds(e): (e / "preds/round_002.u8.npy").unlink()
    def m_nologs(e): (e / "logs/round_002.json").unlink()
    def m_lr(e):
        lg = _rl(e); lg["clients"][0]["lr"] = cfg["lr"]; _wl(e, lg)
        j = _rj(e); j["clients"] = lg["clients"]; _wj(e, j)
    def m_lr_row(e):
        j = _rj(e); j["lr"] = cfg["lr"]; _wj(e, j)
    def m_nparams(e):
        j = _rj(e); j["n_params"] = 395_024; _wj(e, j)
    def m_weight_nan(e):
        w = torch.load(e / "weights/round_002.pt", map_location="cpu", weights_only=True)
        k = next(iter(w["global"])); w["global"][k] = w["global"][k].clone(); w["global"][k].view(-1)[0] = float("nan")
        C.atomic_save(w, e / "weights/round_002.pt")
    def m_short(e): (e / "complete/round_002.done").unlink()
    def m_no_cm(e): (e / "confusion/round_002.npy").unlink()
    def m_resume_round(e):
        r = torch.load(e / "resume/round_002.pt", map_location="cpu", weights_only=True)
        r["round"] = 7; C.atomic_save(r, e / "resume/round_002.pt")
    for label, fn in (("NaN metric", m_nan), ("metric overwritten in json", m_json),
                      ("metric overwritten in the weights file", m_w),
                      ("confusion edited", m_cm),
                      ("history.csv row dropped", m_csv), ("history.csv value edited", m_csv_val),
                      ("clients.csv value edited", m_ccsv),
                      ("per-class f1 = 999", m_pc),
                      ("client missing from the log", m_log_missing_client),
                      ("logs/ and metrics/ per-client blocks disagree", m_log_json_disagree),
                      ("client applied 0 steps", m_steps),
                      ("client applied a non-finite step", m_nonfin),
                      ("client mean overwritten", m_client_mean),
                      ("round step total overwritten", m_steps_total),
                      ("predictions edited", m_preds),
                      ("claimed predictions missing", m_nopreds), ("client log missing", m_nologs),
                      ("client trained at a rate off the schedule", m_lr),
                      ("row lr off the schedule", m_lr_row),
                      ("row n_params off the cfg", m_nparams),
                      ("NaN in a weight", m_weight_nan), ("run shorter than required", m_short),
                      ("confusion file missing", m_no_cm),
                      ("resume file claims another round", m_resume_round)):
        case(label, fn)

    # ---- resume import from a READ-ONLY attached source, markers bound the import
    src_root = tmp / "input" / "prev"
    src = src_root / "runs" / "r"
    for s in C.SUBDIRS: (src / s).mkdir(parents=True, exist_ok=True)
    for s in C.SUBDIRS:
        for f in (d / s).iterdir():
            if f.is_file(): shutil.copyfile(f, src / s / f.name)
    (src / "complete/round_002.done").unlink()               # source committed only round 1
    for p in src.rglob("*"): os.chmod(p, 0o444 if p.is_file() else 0o555)
    dest = mk(tmp / "runs2" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs2" / name)
    last = C.resolve_resume("r", cfg, attached=tmp / "input")
    assert last == 1, last
    assert (dest / "weights/round_001.pt").is_file() and not (dest / "complete/round_002.done").exists()
    assert (dest / "confusion/round_001.npy").is_file() and (dest / "logs/round_001.json").is_file()
    assert os.access(dest / "weights/round_001.pt", os.W_OK)
    rows = (dest / "history.csv").read_text().splitlines()
    assert len(rows) == 2, rows
    print("5. import from a read-only mount: only the source's committed round 1, writable, "
          "history rebuilt"); n += 1

    # a second, different history with the same fingerprint must be refused
    other = mk(tmp / "runs3" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs3" / name)
    write_run(other, cfg, seed=1)
    try:
        C.resolve_resume("r", cfg, attached=tmp / "input"); raise AssertionError("spliced")
    except RuntimeError as e:
        assert "different weights" in str(e)
    print("6. two histories with one fingerprint: refused, not spliced"); n += 1

    for p in src_root.rglob("*"): os.chmod(p, 0o644 if p.is_file() else 0o755)

    # ---- handoff bundle: round r alone + the sha chain 1..r, anchored both ways
    full = mk(tmp / "full" / "runs" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "full" / "runs" / name)
    write_run(full, {**cfg, "rounds": 3}, seed=2, rounds=3)
    cfg3 = {**cfg, "rounds": 3}; fp3 = C.fingerprint(cfg3)
    try: C.handoff_record(full, 3, "wrong"); raise AssertionError
    except RuntimeError: pass
    rec = C.handoff_record(full, 2, fp3)
    assert rec["round"] == 2 and sorted(rec["chain"]) == ["1", "2"]
    assert rec["chain"]["2"] == C.file_sha(full / "weights/round_002.pt")

    def bundle(dst, rnd, rec):
        for s in C.SUBDIRS: (dst / s).mkdir(parents=True, exist_ok=True)
        for s in C.RESUME_SUBDIRS + ("complete",):
            for f in (full / s).glob(f"*_{rnd:03d}.*"): shutil.copyfile(f, dst / s / f.name)
        (dst / "reports/manifest.json").write_text((full / "reports/manifest.json").read_text())
        if rec is not None: (dst / C.HANDOFF).write_text(json.dumps(rec))
        return dst
    b = bundle(tmp / "bundle" / "runs" / "r", 2, rec)
    assert C._first_round(b) == 2 and C.last_complete_round(b, fp3) == 2
    dest = mk(tmp / "runs4" / "r")
    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs4" / name)
    last = C.resolve_resume("r", cfg3, attached=tmp / "bundle")
    assert last == 2, last
    assert (dest / "weights/round_002.pt").is_file() and not (dest / "weights/round_001.pt").exists()
    assert (dest / C.HANDOFF).is_file()
    rows = (dest / "history.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("2,"), rows
    (dest / "reports/manifest.json").write_text((full / "reports/manifest.json").read_text())
    ok, lines = verify_run(dest, cfg=cfg3, build_model=build_model, expect_params=cfg["n_params"],
                           y_true_path=full / "reports/y_true.u8.npy", require_rounds=2)
    assert ok and any("rounds 2..2" in l for l in lines), "\n".join(lines)
    print("7. handoff bundle (round 2 + chain): imports as round 2 alone, history from 2, verifies"); n += 1

    for label, rec2 in (("chain[2] wrong", {**rec, "chain": {**rec["chain"], "2": "0" * 64}}),
                        ("chain[1] wrong", {**rec, "chain": {**rec["chain"], "1": "0" * 64}}),
                        ("round mismatch", {**rec, "round": 3}),
                        ("no handoff", None)):
        shutil.rmtree(tmp / "bundle2", ignore_errors=True)
        b2 = bundle(tmp / "bundle2" / "runs" / "r", 2, rec2)
        assert C.last_complete_round(b2, fp3) is None, label
        C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs5" / name)
        shutil.rmtree(tmp / "runs5", ignore_errors=True)
        assert C.resolve_resume("r", cfg3, attached=tmp / "bundle2") is None, label
    print("8. tampered chain / wrong round / missing handoff: refused, nothing imported"); n += 1

    C.run_dir = lambda name, base=None, _t=tmp: mk(_t / "runs6" / name)
    write_run(mk(tmp / "runs6" / "r"), cfg3, seed=3, rounds=1)
    try: C.resolve_resume("r", cfg3, attached=tmp / "bundle"); raise AssertionError("spliced")
    except RuntimeError as e: assert "own history" in str(e)
    print("9. handoff onto a tree with its own rounds: refused"); n += 1

    # ---- merge: s1 (rounds 1..2) + s2 (handoff round 2 + round 3) == the single-session tree
    import filecmp, subprocess
    s1 = tmp / "pull_s1" / "runs" / "r"
    for sub in C.SUBDIRS: (s1 / sub).mkdir(parents=True, exist_ok=True)
    for sub in C.RESUME_SUBDIRS + ("complete",):
        for f in (full / sub).iterdir():
            if f.is_file() and "_003" not in f.name:
                shutil.copyfile(f, s1 / sub / f.name)
    (s1 / "reports/manifest.json").write_text((full / "reports/manifest.json").read_text())
    C.rebuild_history(s1, fp3)
    assert C.last_complete_round(s1, fp3) == 2
    s2 = bundle(tmp / "pull_s2" / "runs" / "r", 2, rec)
    for sub in C.RESUME_SUBDIRS + ("complete",):
        for f in (full / sub).glob("*_003.*"): shutil.copyfile(f, s2 / sub / f.name)
    C.rebuild_history(s2, fp3)
    assert C.last_complete_round(s2, fp3) == 3 and C._first_round(s2) == 2
    merged = tmp / "merged"
    MERGE = Path(__file__).resolve().parents[2] / ".claude/skills/kaggle-training-notebook/scripts/merge_sessions.py"
    r = subprocess.run([sys.executable, str(MERGE), "--out", str(merged),
                        str(tmp / "pull_s1"), str(tmp / "pull_s2")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for sub in C.RESUME_SUBDIRS + ("complete",):
        for f in (full / sub).iterdir():
            if f.is_file():
                assert filecmp.cmp(f, merged / sub / f.name, shallow=False), f"{sub}/{f.name}"
    assert not (merged / C.HANDOFF).exists() and (merged / "logs/sessions/2/handoff.json").is_file()
    assert C._first_round(merged) == 1 and C.last_complete_round(merged, fp3) == 3
    hrows = (merged / "history.csv").read_text().splitlines()
    assert [l.split(",")[0] for l in hrows[1:]] == ["1", "2", "3"], hrows
    assert (merged / "history.csv").read_text() == (full / "history.csv").read_text()
    assert (merged / "clients.csv").read_text() == (full / "clients.csv").read_text()
    ok, lines = verify_run(merged, cfg=cfg3, build_model=build_model, expect_params=cfg["n_params"],
                           y_true_path=full / "reports/y_true.u8.npy", require_rounds=3)
    assert ok and any("rounds 1..3" in l for l in lines), "\n".join(lines)
    s2b = tmp / "pull_s2b"; shutil.copytree(tmp / "pull_s2", s2b)
    (s2b / "runs/r/weights/round_003.pt").write_bytes(b"not the same bytes")
    r = subprocess.run([sys.executable, str(MERGE), "--out", str(tmp / "merged2"),
                        str(tmp / "pull_s1"), str(tmp / "pull_s2"), str(s2b)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "differs between sessions" in (r.stdout + r.stderr)
    print("10. merge s1 + handoff s2: byte-identical to the one-session tree, history/clients "
          "1..3, verifies; a retrained overlap is refused"); n += 1

    shutil.rmtree(tmp)
    print(f"\nALL {n} CHECKPOINT / VERIFIER CHECKS PASSED")


if __name__ == "__main__":
    main()
