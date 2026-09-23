"""Schedule independence: 2 CPU workers (dynamic LPT dispatch, the test set split by row
range between them) must produce the SAME weights and metrics as 1 worker, bit for bit.
Small fixture so two worker processes plus the parent stay under the 8 GB watchdog."""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
import torch.multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/lwfednids-bouayad-2024"))
import tests.test_smoke_real as S
from proj import ckpt as C
from proj import driver as D

S.ROWS_PER_CLIENT, S.TEST_ROWS = 2_000, 4_000


def _prepack_child(cache, cids):
    spans, n_test = S.prepack(cache, cids)
    json.dump({"spans": {str(k): v for k, v in spans.items()}, "n_test": n_test},
              open(cache / "spans.json", "w"))


def prepack(cache, cids):
    """In a child process: the parent must never import pyarrow, since it stays resident
    next to two torch workers and the three together must fit the 8 GB watchdog."""
    p = mp.get_context("spawn").Process(target=_prepack_child, args=(cache, cids))
    p.start(); p.join(); assert p.exitcode == 0, p.exitcode
    j = json.load(open(cache / "spans.json"))
    return {int(k): tuple(v) for k, v in j["spans"].items()}, j["n_test"]


def run_with(world_size, tmp):
    cache = tmp / f"cache{world_size}"; cache.mkdir()
    cids = [0, 1, 2, 3]
    spans, n_test = prepack(cache, cids)
    cfg = S.make_cfg(cache, len(cids), n_test, rounds=1, world_size=world_size)
    cfg["run_name"] = f"w{world_size}"
    cfg["batch"], cfg["eval_batch"] = 64, 256        # activations, not data, set the RSS here
    C.run_dir = lambda name, base=None, _t=tmp: S._mk(_t / "runs" / name)
    D.write_manifest(cfg, S.META["class_names"], spans, y_true_src=cache / "test_y.u8.npy",
                     extra={"n_test": n_test, "content_id": "smoke_content"})
    hist = D.run(cfg, spans, S.META["class_names"])
    assert len(hist) == 1
    d = tmp / "runs" / cfg["run_name"]
    return (hist[0], torch.load(d / "weights/round_001.pt", map_location="cpu", weights_only=True),
            np.load(d / "confusion/round_001.npy"), np.load(d / "preds/round_001.u8.npy"))


def main():
    tmp = Path(tempfile.mkdtemp())
    h1, w1, cm1, p1 = run_with(1, tmp)
    h2, w2, cm2, p2 = run_with(2, tmp)
    dmax = max((w1["global"][k].float() - w2["global"][k].float()).abs().max().item()
               for k in w1["global"])
    assert dmax == 0.0, f"1 vs 2 workers differ by {dmax}"
    assert (cm1 == cm2).all() and (p1 == p2).all(), "sharded eval != single-worker eval"
    for k in ("f1_macro", "accuracy", "f1_weighted", "loss_client_mean", "gnorm_client_mean",
              "loss_client_std", "steps", "skipped"):
        assert abs(h1[k] - h2[k]) < 1e-12, (k, h1[k], h2[k])
    assert h1["evaluated"] if "evaluated" in h1 else True
    shutil.rmtree(tmp)
    print(f"\n1 worker vs 2 workers: max|dweight| = {dmax}, confusion + preds identical, "
          f"f1_macro {h1['f1_macro']:.6f} == {h2['f1_macro']:.6f}\nTWO-WORKER TEST PASSED")


if __name__ == "__main__":
    main()
