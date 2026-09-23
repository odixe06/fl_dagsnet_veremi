"""Lightweight-Fed-NIDS (Bouayad, Alami, Janati Idrissi, Berrada — IEEE Access 2024):
the federated loop of §III-B, Algorithm 5 (client) and Algorithm 6 (server), with the
zero-shot pruning mask of §III-B-1-b computed once by the server (proj/prune.py).

Initialization (server, once):
    theta_0  <- seeded DAGSNet                                       §III-B-1-a
    M        <- zero-shot structured mask, L1 group importance      Eq. (7)
    theta'   <- M ⊙ theta_0, pruned channels physically removed     Eq. (8)
    broadcast (theta', M) to every client                           Alg. 6 line 4

Round t, EVERY client j (N = |C|, full participation as in Alg. 6 lines 6-8):
    theta_j <- theta                                                 Alg. 5 line 4
    (the mask is already baked into the architecture: Alg. 5 line 5 is the identity here,
     since a channel that was removed cannot regrow)
    one local epoch over D_j, per batch b:
        L      = CE(f(theta_j, x_b), y_b)                            Eq. (5)/(6), C = 16
        theta_j <- theta_j - eta_t * step(grad L)                    Alg. 5 line 8
    upload theta_j

Server:
    theta^{t+1} = (1/N) sum_{j=1}^{N} theta_j^t                       Eq. (9), Alg. 6 line 9

Deviations from the paper, all decided with the owner on 2026-09-22 and published with
the numbers (docs/REBUILD.md):
  * no feature-extraction backbone (ResNet-50/101, VGG-19 on 40x300-byte flow images):
    the 66 tabular VeReMi features enter the DAGSNet classifier directly;
  * the aggregate is the paper's UNWEIGHTED mean 1/N (Eq. 9), although this partition is
    non-IID with client sizes 870 k .. 5.9 M rows (the paper's IID shards were equal);
  * the local step is AdamW (weight decay 1e-4, knowledge/ARCHITECTURE.md) at a per-ROUND
    cosine learning rate 1e-3 -> 1e-5 over the 50 rounds, constant within a round and
    re-created per client per round, where Algorithm 5 line 8 writes plain gradient
    descent and the paper names no rate;
  * one sparsity level, 0.7 (the paper reports 0 / 0.5 / 0.7 / 0.9);
  * 50 rounds x 1 local epoch, batch 512 / 512 / 256 for 20 / 50 / 100 clients, seed 42.
"""
import contextlib
import math
import torch
import torch.nn.functional as F


def amp(cfg):
    """fp16 autocast on CUDA; a no-op on CPU so the same code runs in the local
    simulation. Never bf16: the T4 is sm_75 and falls back to a slow emulation path."""
    if cfg.get("device", "cuda") == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------- flat layout
# One flat float vector + one int vector per model. A DAGSNet state_dict has 192 entries;
# torch.multiprocessing gives each tensor its own shared-memory fd, so 100 clients a round
# would exhaust the process fd limit. Parameters come FIRST so vec[:n_params] is exactly
# the learnable block; BN running stats follow as buffers.
def layout(model):
    pnames = {n for n, _ in model.named_parameters()}
    sd = model.state_dict()
    fkeys = [k for k in sd if k in pnames]
    fkeys += [k for k in sd if k not in pnames and sd[k].is_floating_point()]
    ikeys = [k for k in sd if not sd[k].is_floating_point()]
    n_params = sum(sd[k].numel() for k in fkeys if k in pnames)
    return fkeys, ikeys, n_params


def flatten(model, fkeys, ikeys):
    sd = model.state_dict()
    fv = torch.cat([sd[k].reshape(-1).float() for k in fkeys])
    iv = torch.stack([sd[k].reshape(-1).long().squeeze() for k in ikeys]) if ikeys \
        else torch.zeros(0, dtype=torch.long)
    return fv, iv


def unflatten_into(model, fv, iv, fkeys, ikeys):
    sd = model.state_dict()
    o = 0
    for k in fkeys:
        t = sd[k]; n = t.numel()
        t.copy_(fv[o:o + n].view_as(t)); o += n           # copy_ keeps addresses -> CUDA graph valid
    for j, k in enumerate(ikeys):
        sd[k].copy_(iv[j].view_as(sd[k]))
    return model


# --------------------------------------------------------------------- learning rate
LR_SCHEDULES = ("constant", "cosine")


def lr_at(cfg, rnd):
    """Learning rate of round `rnd` (1-based), held constant within the round. A pure
    function of (cfg, rnd): a resumed session applies exactly the value the original
    session would have.

      constant : cfg['lr'] every round
      cosine   : lr_min + (lr - lr_min)/2 * (1 + cos(pi * (rnd-1) / (rounds-1)))
                 -- cfg['lr'] at round 1, cfg['lr_min'] at round cfg['rounds'].
    """
    sched = cfg.get("lr_schedule", "constant")
    if sched == "constant":
        return float(cfg["lr"])
    if sched == "cosine":
        T = int(cfg["rounds"])
        if not 1 <= rnd <= T:
            raise ValueError(f"round {rnd} outside 1..{T}: the cosine schedule is undefined")
        if T == 1:
            return float(cfg["lr"])
        lo, hi = float(cfg["lr_min"]), float(cfg["lr"])
        return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * (rnd - 1) / (T - 1)))
    raise ValueError(f"lr_schedule {sched!r} not in {LR_SCHEDULES}")


# --------------------------------------------------------------------- the loss
# Order of the per-step values accumulated on the device (see `client_update`).
ACC_KEYS = ("loss", "gnorm")


def ce_loss(z, y):
    """Eq. (5)/(6) for one batch, from fp32 logits: the categorical cross-entropy of the
    softmax over the 16 classes (the paper's binary form is the C = 2 case)."""
    return F.cross_entropy(z, y)


# --------------------------------------------------------------------- client update
def make_optimizer(Se, lr, cfg, fused):
    """AdamW over the (pruned) model's parameters, weight decay from cfg. fused=True
    collapses the step into one multi-tensor kernel on CUDA."""
    return torch.optim.AdamW(list(Se.parameters()), lr=lr,
                             weight_decay=cfg["weight_decay"], fused=fused)


def client_update(Sc, Se, opt, scaler, X, Y, lo, hi, cfg, gen):
    """Algorithm 5 lines 6-9: `local_epochs` passes over rows [lo, hi) of the resident
    tensors for one client.

    Se is the eager module that owns theta_j; Sc the compiled alias (or the same object
    when compile is off). The tail batch is a different shape and would recompile the
    CUDA graph once per client, so it runs on the eager module: same weights, same math.

    Returns (acc, n_steps): `acc` is a device tensor of len(ACC_KEYS) + 2 sums over
    APPLIED steps (the last two entries are the skipped-step count and the count of
    applied steps whose gradient norm was not finite), read once by the caller.
    Dropout reads torch's default generator, which the worker re-seeds from
    (seed, round, client) before calling this; `gen` drives only the shuffles."""
    B, clip, dev = cfg["batch"], cfg["clip"], X.device
    params = list(Se.parameters())
    Se.train()
    acc = torch.zeros(len(ACC_KEYS) + 2, device=dev)
    zeros = torch.zeros(len(ACC_KEYS), device=dev)
    n = 0
    for _ in range(cfg["local_epochs"]):
        perm = lo + torch.randperm(hi - lo, generator=gen, device=dev)
        for i in range(0, hi - lo, B):
            # One captured graph per step; the explicit iteration boundary keeps CUDA-graph
            # trees from treating the next replay as part of this one.
            if X.is_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            idx = perm[i:i + B]
            full = idx.numel() == B
            xb = X[idx].float()
            yb = Y[idx].long()
            with amp(cfg):
                z = (Sc if full else Se)(xb)
            loss = ce_loss(z.float(), yb)                          # loss in fp32
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                   # grads now in true units
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            prev = scaler._scale.clone() if scaler.is_enabled() else None
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            # A skipped step overflowed: its grad-norm is inf and its loss may be nan.
            # torch.where, not multiplication -- inf*0 is nan.
            applied = (scaler._scale >= prev) if prev is not None \
                else torch.ones((), dtype=torch.bool, device=dev)
            vals = torch.stack([loss.detach(), gn])
            acc[:len(ACC_KEYS)] += torch.where(applied, vals, zeros)
            acc[len(ACC_KEYS)] += (~applied).float()
            acc[len(ACC_KEYS) + 1] += ((~torch.isfinite(gn)) & applied).float()
            n += 1
    return acc, n


def expected_steps(n_k, cfg):
    return cfg["local_epochs"] * math.ceil(n_k / cfg["batch"])


# --------------------------------------------------------------------- server side
def aggregate(updates, n_clients):
    """Eq. (9) / Algorithm 6 line 9: theta^{t+1} = (1/N) sum_{j=1}^N theta_j^t.

    UNWEIGHTED, as the paper writes it -- a client with 870 k rows counts exactly as much
    as one with 5.9 M (docs/REBUILD.md). `updates` must hold every one of the N clients,
    sorted by client id: float addition order decides the result, and it must not be set
    by a completion race. A missing client is an error, never a smaller divisor: dropping
    one changes the participation the paper fixes at all N."""
    cids = [c for c, _, _ in updates]
    if cids != list(range(n_clients)):
        raise ValueError(f"aggregate() needs every client 0..{n_clients - 1} in order, "
                         f"got {cids[:6]}...")
    acc = None
    for cid, fv, iv in updates:
        acc = fv / n_clients if acc is None else acc.add_(fv, alpha=1.0 / n_clients)
    # num_batches_tracked is an int counter, not an averageable quantity; with the default
    # BatchNorm momentum=0.1 it is unused at inference. Take the max so it stays monotone.
    ints = torch.stack([iv for _, _, iv in updates]).amax(dim=0) if updates[0][2].numel() \
        else updates[0][2]
    return acc, ints
