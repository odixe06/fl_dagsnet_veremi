"""Real torch.compile(reduce-overhead) on the local GPU (sm_86) with the PRUNED model: the
train step under CUDA graphs, graph reuse across clients, tail-batch fallback, AMP skip
accounting, compiled-vs-eager agreement, and the eval template. Also measures the
unpruned model's step next to the pruned one's, since the paper's claim is a speed-up.
sm_86 passing does NOT prove sm_75; it proves the graph scheme is sound."""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/lwfednids-bouayad-2024"))
import tests.test_smoke_real as S
import tests.test_two_workers as W          # prepack in a child: no pyarrow next to Inductor
from proj.model import build_model, n_params
from proj.lwfednids import (layout, flatten, unflatten_into, client_update, make_optimizer,
                            ACC_KEYS)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj import driver as D

S.ROWS_PER_CLIENT, S.TEST_ROWS = 3_000, 12_000


def train_one(Sc, Se, X, Y, lo, hi, cfg, dev, seed=7):
    opt = make_optimizer(Se, cfg["lr"], cfg, fused=True)
    sc = torch.amp.GradScaler("cuda"); sc.scale(torch.zeros(1, device=dev))
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    acc, n = client_update(Sc, Se, opt, sc, X, Y, lo, hi, cfg, g)
    torch.cuda.synchronize()
    return acc.cpu().tolist(), n, (time.perf_counter() - t0) / n * 1000


def main():
    assert torch.cuda.is_available(), "needs the local GPU"
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    for _n in ('recompile_limit', 'cache_size_limit'):
        if hasattr(torch._dynamo.config, _n): setattr(torch._dynamo.config, _n, 64)
    tmp = Path(tempfile.mkdtemp()); cache = tmp / "cache"; cache.mkdir()
    spans, n_test = W.prepack(cache, [0, 1])
    cfg = S.make_cfg(cache, 2, n_test, rounds=1)
    cfg.update(device="cuda", compile=True, batch=256, eval_batch=4096)
    X = D._resident(cache / "train_X.f16.npy", dev); Y = D._resident(cache / "train_y.u8.npy", dev)
    TX = D._resident(cache / "test_X.f16.npy", dev); TY = D._resident(cache / "test_y.u8.npy", dev)
    B = cfg["batch"]
    K = len(ACC_KEYS)

    torch.manual_seed(42)
    Se = build_model(cfg).to(dev).train()
    assert n_params(Se) == cfg["n_params"] == 35_891
    fk, ik, _ = layout(Se)
    s0, i0 = flatten(Se, fk, ik)

    def reset():
        unflatten_into(Se, s0.to(dev), i0.to(dev), fk, ik)

    # 1. the real gate on real rows
    Sc = D._compile_train(Se, cfg, dev, X[:B].float(), Y[:B].long())
    assert Sc is not Se, "compile gate fell back to eager on sm_86"
    print("1. train gate certified with real torch.compile(reduce-overhead) on the pruned model")

    # 2. two clients back to back on the compiled module (graph reuse, tail batch, AMP)
    for cid in (0, 1):
        reset()
        a, n, ms = train_one(Sc, Se, X, Y, *spans[cid], cfg, dev)
        sk = int(a[K]); ap = max(1, n - sk)
        st = dict(cid=cid, steps=n, skips=sk, loss=a[0] / ap, gnorm=a[1] / ap, ms_per_step=round(ms, 2))
        assert n == 12 and all(np.isfinite([st["loss"], st["gnorm"]])), st
        assert int(a[K + 1]) == 0
        sv, _ = flatten(Se, fk, ik)
        assert torch.isfinite(sv).all()
        print(f"2. client {cid}: {st}")
    comp_s = flatten(Se, fk, ik)[0].clone()

    # 3. compiled vs eager on the same client from the same weights with dropout off:
    #    the two paths must land within fp16 accumulation distance of each other
    for m in Se.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.0
    outs = []
    for Sm in (Sc, Se):
        reset()
        a, n, ms = train_one(Sm, Se, X, Y, *spans[1], cfg, dev)
        outs.append((flatten(Se, fk, ik)[0].clone(), a[0] / max(1, n - int(a[K])), ms))
    ds = (outs[0][0] - outs[1][0]).abs().max().item()
    rel = ds / (outs[1][0] - s0.to(dev)).abs().max().item()
    print(f"3. compiled vs eager (dropout 0): max|dw|={ds:.2e} "
          f"(rel. to the update {rel:.2e}) | loss {outs[0][1]:.4f} vs {outs[1][1]:.4f} | "
          f"{outs[0][2]:.2f} vs {outs[1][2]:.2f} ms/step -> {outs[1][2]/outs[0][2]:.2f}x")
    assert rel < 0.5 and abs(outs[0][1] - outs[1][1]) / outs[1][1] < 5e-2, (rel, outs[0][1], outs[1][1])
    for m in Se.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.1

    # 4. eval template: compile, gate, run on the whole test with a tail batch
    Te = fold_bn(build_model(cfg).to(dev))
    Tc = D._compile_eval(Te, cfg, dev, TX[:cfg["eval_batch"]].float())
    assert Tc is not Te, "eval gate fell back to eager on sm_86"
    unflatten_into(Se, comp_s, i0.to(dev), fk, ik)
    load_folded(Te, Se)
    eval_model(Tc, Te, TX, TY, cfg)          # first pass pays the one-off graph warm-up
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cm_c, nf, preds = eval_model(Tc, Te, TX, TY, cfg, want_preds=True)
    torch.cuda.synchronize(); tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    cm_e, nf2, preds2 = eval_model(Te, Te, TX, TY, cfg, want_preds=True)
    torch.cuda.synchronize(); te = time.perf_counter() - t0
    assert cm_c.sum().item() == n_test == cm_e.sum().item() and nf == nf2 == 0
    dis = int((preds != preds2).sum())
    print(f"4. eval: compiled {n_test/tc:,.0f} rows/s vs eager-folded {n_test/te:,.0f} rows/s "
          f"({te/tc:.2f}x); {dis} of {n_test} predictions differ ({dis/n_test:.4%})")
    assert dis / n_test < 1e-3
    assert tc < te, "compiled eval slower than eager-folded in steady state: CUDA graphs re-recording"
    # sharded eval on the compiled template equals the full pass
    h = n_test // 2
    ca, _, pa = eval_model(Tc, Te, TX[:h], TY[:h], cfg, want_preds=True)
    cb, _, pb = eval_model(Tc, Te, TX[h:], TY[h:], cfg, want_preds=True)
    d_sh = int(((ca + cb) - cm_c).abs().sum())
    print(f"5. sharded compiled eval: |dCM| = {d_sh} cells vs the single pass")
    assert d_sh == 0 and torch.equal(torch.cat([pa, pb]), preds)

    # 6. the unpruned model through the same paths, for the speed-up the paper claims
    cfg0 = dict(cfg, sparsity=0.0, prune_plan=None, plan_id=None, n_params=395_024)
    torch.manual_seed(42); F = build_model(cfg0).to(dev).train()
    fk0, ik0, _ = layout(F); f0, fi0 = flatten(F, fk0, ik0)
    torch._dynamo.reset()
    Fc = D._compile_train(F, cfg0, dev, X[:B].float(), Y[:B].long())
    assert Fc is not F
    unflatten_into(F, f0, fi0, fk0, ik0); train_one(Fc, F, X, Y, *spans[1], cfg0, dev)
    unflatten_into(F, f0, fi0, fk0, ik0); _, _, ms_full = train_one(Fc, F, X, Y, *spans[1], cfg0, dev)
    reset(); train_one(Sc, Se, X, Y, *spans[1], cfg, dev)
    reset(); _, _, ms_pruned = train_one(Sc, Se, X, Y, *spans[1], cfg, dev)
    print(f"6. compiled step at batch {B}: unpruned {ms_full:.2f} ms vs pruned {ms_pruned:.2f} ms "
          f"-> TA = {ms_full/ms_pruned:.2f}x (sm_86; measure again on the T4)")
    print(f"   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    shutil.rmtree(tmp)
    print("\nLOCAL GPU COMPILE TEST PASSED (sm_86; not evidence for sm_75)")


if __name__ == "__main__":
    main()
