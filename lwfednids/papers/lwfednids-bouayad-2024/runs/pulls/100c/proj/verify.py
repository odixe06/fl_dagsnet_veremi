"""Re-derive every published number from the artifacts on disk.

Nothing here trusts a number because it was printed once. Per round, the 10 metrics and
the per-class block are recomputed from the global model's confusion matrix; the means /
std / min / max over clients of the training statistics are recomputed from the per-client
log; predictions, where stored, rebuild the confusion matrix; the client log is checked
against the step arithmetic and the schedule it claims; the weights file must rebuild the
(pruned) model at the parameter count its own cfg declares; and all of it is compared
against the copies in the weights file, the metrics json, history.csv and clients.csv.

Every check exists because its absence lets a specific tampered fixture pass: a deleted
history row, a duplicated one, a metric set to NaN (`abs(nan) > tol` is False), a
per-class F1 of 999, deleted logs, a client missing from the log, a metric overwritten in
one of its four copies, a resume file overwritten with garbage, a config claiming 999
test rows, a plan whose parameter count is not the one the weights were trained at.
"""
import csv, json, math
from pathlib import Path
import numpy as np
import torch

from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
from proj.lwfednids import expected_steps, lr_at, ACC_KEYS
from proj import ckpt as C

TOL = 1e-12          # both sides come from the same float64 code path on the same counts
CSV_TOL = 1e-9       # history.csv round-trips through str()
CLIENT_INT = ("cid", "n_k", "steps", "applied", "skipped", "nonfinite")
CLIENT_FLOAT = ("lr",) + ACC_KEYS


def _finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def verify_run(run_dir, cfg=None, build_model=None, expect_params=None, y_true_path=None,
               require_rounds=None, full=True):
    """Returns (ok, lines). full=True is the acceptance mode: client logs must be present
    for every round and predictions for every round that claims them."""
    d = Path(run_dir)
    fp = C.fingerprint(cfg) if cfg is not None else None
    last = C.last_complete_round(d, fp) or 0
    # A tree that starts past round 1 is a session resumed from a handoff bundle: rounds
    # before `first` are attested by reports/handoff.json (checked inside round_ok) and are
    # re-verified in full only after the sessions are merged. Nothing here is skipped for
    # the rounds that ARE on disk.
    first = C._first_round(d) if last else 1
    mode = "full" if full else "minimal"
    out = [f"run      : {d}", f"complete : rounds {first}..{last}   (mode: {mode})"
           + (f"   [handoff bundle: rounds 1..{first} attested by {C.HANDOFF}]" if first > 1 else "")]
    bad = []

    mf = d / "reports" / "manifest.json"
    man = json.loads(mf.read_text()) if mf.is_file() else None
    if man is None and full:
        bad.append("reports/manifest.json missing: the run does not describe itself")

    N = int(cfg["n_clients"]) if cfg else (int(man["n_clients"]) if man else None)
    n_test = int(cfg["n_test"]) if cfg and "n_test" in cfg else None
    want_rows = ({int(k): int(v) for k, v in man["client_rows"].items()}
                 if man and "client_rows" in man else None)
    want_params = int(cfg["n_params"]) if cfg else (int(man["n_params"]) if man else None)

    # ---- history.csv / clients.csv: exactly rounds first..last, once each
    hist, hp = {}, d / "history.csv"
    if hp.is_file():
        rows = _read_csv(hp)
        seen = [int(r["round"]) for r in rows]
        if len(seen) != len(set(seen)):
            bad.append(f"history.csv has duplicate rows for round(s) "
                       f"{sorted({r for r in seen if seen.count(r) > 1})}")
        if sorted(set(seen)) != list(range(first, last + 1)):
            bad.append(f"history.csv covers rounds {sorted(set(seen))}, expected {first}..{last}")
        hist = {int(r["round"]): r for r in rows}
    elif last:
        bad.append("history.csv missing")
    crows, cp = {}, d / "clients.csv"
    if cp.is_file():
        for r in _read_csv(cp):
            crows.setdefault(int(r["round"]), {})[int(r["cid"])] = r
    elif last:
        bad.append("clients.csv missing")

    y_true = None
    if y_true_path is not None and Path(y_true_path).is_file():
        y_true = np.load(y_true_path, mmap_mode="r")

    for r in range(first, last + 1):
        tag = f"round {r:03d}"
        cm = np.load(d / "confusion" / f"round_{r:03d}.npy")
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            bad.append(f"{tag}: confusion is {cm.shape}, expected (C, C)"); continue
        if (cm < 0).any():
            bad.append(f"{tag}: negative counts in the confusion matrix")
        if cfg and cm.shape[0] != int(cfg["num_classes"]):
            bad.append(f"{tag}: {cm.shape[0]} classes, cfg says {cfg['num_classes']}")
        if n_test is None: n_test = int(cm.sum())
        if cm.sum() != n_test:
            bad.append(f"{tag}: confusion total {int(cm.sum())} != n_test {n_test}")

        js = json.loads((d / "metrics" / f"round_{r:03d}.json").read_text())
        ck = torch.load(d / "weights" / f"round_{r:03d}.pt", map_location="cpu",
                        weights_only=True, mmap=True)
        rs = torch.load(d / "resume" / f"round_{r:03d}.pt", map_location="cpu",
                        weights_only=True)
        if int(rs.get("round", -1)) != r:
            bad.append(f"{tag}: resume file claims round {rs.get('round')}")

        # ---- the global model's 10 metrics + per-class, in json, weights file, csv
        rec = metrics_from_confusion(cm)
        for k in METRIC_KEYS:
            for where, val, tol in (("json", js.get(k), TOL),
                                    ("history.csv", hist.get(r, {}).get(k), CSV_TOL),
                                    ("weights file", ck.get("metrics", {}).get(k), TOL)):
                if val is None:
                    bad.append(f"{tag}: {k} missing in {where}"); continue
                if not _finite(val):
                    bad.append(f"{tag}: {k} in {where} is not finite ({val!r})")
                elif abs(float(val) - rec[k]) > tol:
                    bad.append(f"{tag}: {k} in {where} = {val} != {rec[k]} recomputed")
        names = [e.get("class") for e in js.get("per_class", [])]
        if len(names) != cm.shape[0]:
            bad.append(f"{tag}: per-class block missing or wrong length")
        else:
            for got, want in zip(js["per_class"], per_class_from_confusion(cm, names)):
                for f in ("idx", "support"):
                    if int(got.get(f, -1)) != int(want[f]):
                        bad.append(f"{tag}: class {want['idx']} {f} {got.get(f)} != {want[f]}")
                for f in ("precision", "recall", "f1"):
                    v = got.get(f)
                    if not _finite(v) or abs(float(v) - want[f]) > TOL:
                        bad.append(f"{tag}: class {want['idx']} {f} {v!r} != {want[f]}")
        if cfg is not None:
            want_lr = lr_at(cfg, r)
            if not _finite(js.get("lr")) or abs(float(js["lr"]) - want_lr) > 1e-12:
                bad.append(f"{tag}: json lr {js.get('lr')!r} != schedule {want_lr}")
            if int(js.get("n_params", -1)) != want_params:
                bad.append(f"{tag}: json n_params {js.get('n_params')} != cfg {want_params}")
            if float(js.get("sparsity", -1)) != float(cfg["sparsity"]):
                bad.append(f"{tag}: json sparsity {js.get('sparsity')} != cfg {cfg['sparsity']}")
        if fp is not None and ck.get("fingerprint") != fp:
            bad.append(f"{tag}: weights fingerprint {ck.get('fingerprint')} != {fp}")
        if build_model is not None:
            try:
                C.load_weights(d / "weights" / f"round_{r:03d}.pt", build_model, expect_params)
            except Exception as e:
                bad.append(f"{tag}: weights do not rebuild the model: {e}")

        # ---- predictions tie the matrix back to model output, where stored
        pp = d / "preds" / f"round_{r:03d}.u8.npy"
        claims = cfg is not None and r in set(cfg.get("preds_rounds", [cfg["rounds"]]))
        if pp.is_file():
            yp = np.load(pp, mmap_mode="r")
            if yp.shape != (n_test,):
                bad.append(f"{tag}: predictions are {yp.shape}, expected ({n_test},)")
            elif y_true is not None:
                if len(y_true) != n_test:
                    bad.append(f"{tag}: y_true has {len(y_true)} rows, test has {n_test}")
                else:
                    k = cm.shape[0]
                    rebuilt = np.bincount(np.asarray(y_true, np.int64) * k
                                          + np.asarray(yp, np.int64),
                                          minlength=k * k).reshape(k, k)
                    if not (rebuilt == cm).all():
                        bad.append(f"{tag}: confusion != stored predictions")
            elif full:
                bad.append(f"{tag}: predictions present but no y_true to check them against")
        elif claims and full:
            bad.append(f"{tag}: cfg claims predictions for this round but none are stored")

        # ---- client logs: every client, step arithmetic and the schedule have to close;
        #      the json's copy, clients.csv and the round's client_* aggregates must agree
        lp = d / "logs" / f"round_{r:03d}.json"
        lg_clients = None
        if not lp.is_file():
            if full:
                bad.append(f"{tag}: no client log; participation is unattested")
        else:
            try: lg_clients = json.loads(lp.read_text()).get("clients", [])
            except Exception as e:
                bad.append(f"{tag}: client log unreadable: {e}"); lg_clients = []
        jc = js.get("clients")
        if jc is None:
            bad.append(f"{tag}: metrics json carries no per-client block")
        clients = lg_clients if lg_clients is not None else (jc or [])
        if lg_clients is not None and jc is not None and lg_clients != jc:
            bad.append(f"{tag}: logs/ and metrics/ disagree on the per-client block")
        got_ids = sorted(int(e["cid"]) for e in clients)
        if N is not None and got_ids != list(range(N)):
            bad.append(f"{tag}: log has clients {got_ids[:6]}..., expected all 0..{N - 1}")
        per = {k: [] for k in ACC_KEYS}
        for e in clients:
            c = e.get("cid")
            if e.get("applied", 0) + e.get("skipped", 0) != e.get("steps", -1):
                bad.append(f"{tag}: client {c} applied+skipped != steps")
            if e.get("applied", 0) <= 0:
                bad.append(f"{tag}: client {c} applied no step")
            if e.get("nonfinite", 0):
                bad.append(f"{tag}: client {c} applied a step with a non-finite gradient")
            if cfg and "n_k" in e:
                s = expected_steps(int(e["n_k"]), cfg)
                if int(e.get("steps", -1)) != s:
                    bad.append(f"{tag}: client {c} ran {e.get('steps')} steps, expected {s}")
            if want_rows is not None and c in want_rows and int(e.get("n_k", -1)) != want_rows[c]:
                bad.append(f"{tag}: client {c} trained on {e.get('n_k')} rows, "
                           f"manifest says {want_rows[c]}")
            # The rate the client actually used must be the schedule's value for this
            # round: a resumed session that planned a different horizon would otherwise
            # continue the run at a rate the fingerprint never saw.
            if cfg is not None:
                want_lr = lr_at(cfg, r)
                if not _finite(e.get("lr")) or abs(float(e["lr"]) - want_lr) > 1e-12:
                    bad.append(f"{tag}: client {c} trained at lr {e.get('lr')!r}, "
                               f"schedule says {want_lr}")
            for f in ACC_KEYS:
                if not _finite(e.get(f)):
                    bad.append(f"{tag}: client {c} {f} is not finite ({e.get(f)!r})")
                else:
                    per[f].append(float(e[f]))
            cv = crows.get(r, {}).get(int(c)) if crows else None
            if crows and cv is None:
                bad.append(f"{tag}: client {c} missing from clients.csv")
            elif cv is not None:
                for f in CLIENT_INT:
                    if str(cv.get(f)) != str(e.get(f)):
                        bad.append(f"{tag}: clients.csv client {c} {f} = {cv.get(f)!r} != {e.get(f)!r}")
                for f in CLIENT_FLOAT:
                    if not _finite(cv.get(f)) or abs(float(cv[f]) - float(e[f])) > CSV_TOL:
                        bad.append(f"{tag}: clients.csv client {c} {f} = {cv.get(f)!r} != {e[f]}")
        if clients and len(per[ACC_KEYS[0]]) == len(clients):
            for k in ACC_KEYS:
                v = np.asarray(per[k], dtype=np.float64)
                want = {f"{k}_client_mean": float(v.mean()), f"{k}_client_std": float(v.std()),
                        f"{k}_client_min": float(v.min()), f"{k}_client_max": float(v.max())}
                for kk, wv in want.items():
                    for where, val, tol in (("json", js.get(kk), TOL),
                                            ("history.csv", hist.get(r, {}).get(kk), CSV_TOL)):
                        if val is None:
                            bad.append(f"{tag}: {kk} missing in {where}"); continue
                        if not _finite(val):
                            bad.append(f"{tag}: {kk} in {where} is not finite ({val!r})")
                        elif abs(float(val) - wv) > tol:
                            bad.append(f"{tag}: {kk} in {where} = {val} != {wv} recomputed")
            for f in ("steps", "applied", "skipped"):
                want = sum(int(e.get(f, 0)) for e in clients)
                if int(js.get(f, -1)) != want:
                    bad.append(f"{tag}: json {f} {js.get(f)} != {want} summed over clients")

    n_preds = len(list((d / "preds").glob("round_*.u8.npy"))) if (d / "preds").is_dir() else 0
    n_logs = len(list((d / "logs").glob("round_*.json"))) if (d / "logs").is_dir() else 0
    out.append(f"artifacts: {n_preds} prediction files, {n_logs} client logs"
               + ("" if y_true is not None else "   (predictions NOT cross-checked: no y_true)"))
    if require_rounds is not None and last != require_rounds:
        bad.append(f"run is INCOMPLETE: {last} of {require_rounds} rounds")
    out += [f"  FAIL {b}" for b in bad] or ["  all artifact checks passed"]
    return not bad, out
