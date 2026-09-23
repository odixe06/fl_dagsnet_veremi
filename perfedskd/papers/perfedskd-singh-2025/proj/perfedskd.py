"""PerFed-SKD (Singh, Rupchandani, Adhikari) — the self-knowledge-distillation personalized
FL loop: Eq. (2)-(3), Algorithm 1 (server) and Algorithm 2 (device). The paper's server-side
"Teacher Model trained on a predefined dataset" and any feature-extraction backbone are
deliberately absent; every model in the system is the same DAGSNet classifier
(proj/model.py) and the only knowledge transfer is the one the paper's name refers to —
from a device's OWN previous personalized model to its current one.

Round t, for EVERY client m (Algorithm 2 line 2 iterates over all M in parallel):

    V_m      <- omega_m^{t-1}                 the personalized model saved last round,
                                              FROZEN for the whole round (the teacher)
    omega_m  <- omega^{t-1}  if m in S_t       Algorithm 2 line 5  (selected: take global)
                omega_m^{t-1} otherwise        Algorithm 2 line 7  (keep own weights)

    one local epoch, per batch (x, y):
        z   = f(omega_m, x)                   student, train mode
        z_t = f(V_m, x)                       teacher, eval mode, no grad
        phi = CE(z, y) + lam * KL(p_t || p)                                 Eq. (2)
        omega_m <- omega_m - eta * grad phi                                 Eq. (3)

    upload omega_m^t  <=>  m in S_t           (only the selected devices report back)

Server (Algorithm 1):
    omega^t   = (1/|S_t|) sum_{m in S_t} omega_m^t                          line 10
    tau_t     = (1/M) sum_{m in M} a_m^t                                    line 11
    S_{t+1}   = { m : a_m^t < tau_t }                                       lines 4-6

a_m^t is client m's accuracy on the FIXED GLOBAL TEST SET, the same 10,761,343 rows for
every client (owner's decision, 2026-09-21): the paper says only "local model accuracy"
and a per-client test split does not exist in this partition.

---------------------------------------------------------------------------------------
What the paper leaves open, and what was fixed on 2026-09-21. Every one of these is a
deviation to publish with the numbers; the reasoning is in docs/rebuild.md section 2.

  * `tau`. Algorithm 1 line 11 defines A as the MEAN OF THE CLIENTS' accuracies and line 4
    sets tau = A; the prose of section III-A instead says local accuracy is compared with
    "global aggregated model accuracy". Only the first is written as a formula, and it is
    the only one that selects a subset: on this data the aggregated model beats every
    personalized model on the global test (measured in the sibling rebuild: 0.79 against
    <= 0.70 f1_macro), so the prose reading would select all M clients in every round and
    the method would collapse into FedAvg + SKD. `tau = mean_m a_m` it is.
  * `a_m` is ACCURACY, as the paper writes it, not f1_macro. VeReMi is imbalanced 41:1, so
    accuracy is the weaker statistic here — but it is the paper's rule, and all 10 metrics
    are stored for every client and every round anyway.
  * WHO trains and WHO uploads. Algorithm 1 lines 8-9 loop over S alone; the prose of
    section IV and Algorithm 2 lines 2-7 say every device trains and the unselected ones
    simply keep their own parameters. The prose is followed: all M train, only S upload,
    and the sum in line 10 is over S (which is what its own 1/|S| divisor implies).
  * `L`, the "divergence function" of Eq. (2), is the KL divergence on softmax outputs at
    temperature 1, in the direction KL(teacher || student). `lam` = 1.0. The paper gives
    neither the family, the temperature nor the value.
  * the teacher runs in eval() mode (BatchNorm running statistics, dropout off) and under
    no_grad. A train()-mode teacher would answer with batch statistics and a random
    dropout mask, i.e. a different teacher for every batch.
  * round 1: S_1 = all M (no accuracies exist yet), and V_m = omega^0 for every m, so the
    distillation term starts at exactly 0 and grows as the student leaves its own
    initialization. From round 2 the teacher is a genuinely different model.
  * AdamW, weight decay 1e-4 (knowledge/ARCHITECTURE.md), re-created per client per round,
    so no optimizer state exists at a round boundary and a checkpoint is weights alone.
    The paper says only "Stochastic Gradient Descent" in Eq. (3) and names no rate.
  * the learning rate follows a per-ROUND cosine schedule (`lr_at`), constant within a
    round: the owner's single LR policy across these rebuilds.
"""
import contextlib
import math
import numpy as np
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
ACC_KEYS = ("loss", "ce", "kd", "gnorm")


def skd_loss(z, z_t, y, lam):
    """Eq. (2) for one batch, from fp32 logits.

        ce  = CE(z, y)                          f_m(omega_m^t), the empirical loss
        kd  = KL(p_t || p) = sum p_t (log p_t - log p)      L( x(V_m) || x(omega_m^t) )
        phi = ce + lam * kd

    The teacher's logits arrive already detached (they are produced under no_grad); the
    explicit .detach() here is what keeps that true if a caller ever forgets."""
    ce = F.cross_entropy(z, y)
    log_p = F.log_softmax(z, dim=1)
    log_pt = F.log_softmax(z_t, dim=1).detach()
    # kl_div(input=log q, target=log p, log_target=True) = sum p (log p - log q);
    # batchmean divides by the number of rows, so the scale does not follow the batch.
    kd = F.kl_div(log_p, log_pt, log_target=True, reduction="batchmean")
    return ce + lam * kd, (ce + lam * kd, ce, kd)


# --------------------------------------------------------------------- client update
def make_optimizer(Se, lr, cfg, fused):
    """AdamW over the student's parameters only. The teacher holds no gradient at all:
    it is loaded from the previous round's weights and never stepped."""
    return torch.optim.AdamW(list(Se.parameters()), lr=lr,
                             weight_decay=cfg["weight_decay"], fused=fused)


def client_update(Sc, Se, Vc, Ve, opt, scaler, X, Y, lo, hi, cfg, gen):
    """`local_epochs` passes over rows [lo, hi) of the resident tensors for one client.

    Se is the eager student module that owns omega_m; Ve the eager teacher module holding
    the frozen V_m (named for the paper's V_m -- `Te`/`Tc` mean the folded EVAL template
    elsewhere in this project, and reusing those names here has misled a reader before).
    Sc/Vc are the compiled aliases (or the same objects when compile is off). The tail
    batch is a different shape and would recompile the CUDA graph once per client, so it
    runs on the eager modules: same weights, same math.

    Returns (acc, n_steps): `acc` is a device tensor of len(ACC_KEYS) + 2 sums over
    APPLIED steps (the last two entries are the skipped-step count and the count of
    applied steps whose gradient norm was not finite), read once by the caller.
    Dropout reads torch's default generator, which the worker re-seeds from
    (seed, round, client) before calling this; `gen` drives only the shuffles."""
    B, clip, dev, lam = cfg["batch"], cfg["clip"], X.device, float(cfg["lam"])
    params = list(Se.parameters())
    Se.train()
    Ve.eval()                      # BatchNorm running stats, dropout off: a fixed teacher
    acc = torch.zeros(len(ACC_KEYS) + 2, device=dev)
    zeros = torch.zeros(len(ACC_KEYS), device=dev)
    n = 0
    for _ in range(cfg["local_epochs"]):
        perm = lo + torch.randperm(hi - lo, generator=gen, device=dev)
        for i in range(0, hi - lo, B):
            # Two forward graphs (teacher, student) and one backward run per step.
            # CUDA-graph trees decide which pool memory is dead from the iteration
            # boundary, so declare it explicitly instead of letting the teacher's forward
            # look like a new iteration and invalidate the student's.
            if X.is_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            idx = perm[i:i + B]
            full = idx.numel() == B
            xb = X[idx].float()
            yb = Y[idx].long()
            with torch.no_grad(), amp(cfg):
                z_t = (Vc if full else Ve)(xb)
            z_t = z_t.float()
            with amp(cfg):
                z = (Sc if full else Se)(xb)
            loss, parts = skd_loss(z.float(), z_t, yb, lam)    # loss in fp32
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                               # grads now in true units
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            prev = scaler._scale.clone() if scaler.is_enabled() else None
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            # A skipped step overflowed: its grad-norm is inf and its loss may be nan.
            # torch.where, not multiplication -- inf*0 is nan.
            applied = (scaler._scale >= prev) if prev is not None \
                else torch.ones((), dtype=torch.bool, device=dev)
            vals = torch.stack([p.detach() for p in parts] + [gn])
            acc[:len(ACC_KEYS)] += torch.where(applied, vals, zeros)
            acc[len(ACC_KEYS)] += (~applied).float()
            acc[len(ACC_KEYS) + 1] += ((~torch.isfinite(gn)) & applied).float()
            n += 1
    return acc, n


def expected_steps(n_k, cfg):
    return cfg["local_epochs"] * math.ceil(n_k / cfg["batch"])


# --------------------------------------------------------------------- server side
def aggregate(updates):
    """Algorithm 1 line 10: omega^{t+1} = (1/|S|) sum_{m in S} omega_m^{t+1}.

    `updates` holds ONLY the selected clients and must already be sorted by client id --
    float addition order decides the result, and it must not be set by a completion race.
    An empty S is not aggregated here: the driver keeps the previous global model, since
    a round in which nobody reported has produced no new global information."""
    if not updates:
        raise ValueError("aggregate() called with no selected client; the caller must "
                         "keep the previous global model instead")
    K = len(updates)
    acc = None
    for cid, fv, iv in updates:
        acc = fv / K if acc is None else acc.add_(fv, alpha=1.0 / K)
    # num_batches_tracked is an int counter, not an averageable quantity; with the default
    # BatchNorm momentum=0.1 it is unused at inference. Take the max so it stays monotone.
    ints = torch.stack([iv for _, _, iv in updates]).amax(dim=0) if updates[0][2].numel() \
        else updates[0][2]
    return acc, ints


SELECT_METRIC = "accuracy"          # the paper's a_m / A; see the module docstring


def threshold(acc_by_cid):
    """Algorithm 1 line 11: A <- (1/|M|) sum_{m in M} a_m, over ALL M clients -- including
    the ones that did not upload this round, which is what the sum's |M| divisor says."""
    v = np.asarray([acc_by_cid[c] for c in sorted(acc_by_cid)], dtype=np.float64)
    if v.size == 0:
        raise ValueError("threshold() needs at least one client accuracy")
    return float(v.mean())


def select_clients(acc_by_cid, tau):
    """Algorithm 1 lines 5-6: the devices whose accuracy is BELOW the threshold are the
    ones that receive the new global model. Strictly below, as `if a < tau` is written;
    ties keep their personalized weights."""
    return [c for c in sorted(acc_by_cid) if float(acc_by_cid[c]) < float(tau)]
