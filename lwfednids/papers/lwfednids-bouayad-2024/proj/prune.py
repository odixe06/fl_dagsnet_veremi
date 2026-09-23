"""The pruning mask of Lightweight-Fed-NIDS (Bouayad et al. 2024, §III-B-1-b), computed
ONCE on the server at initialization and never again.

    theta_0  <- seeded initialization of the unpruned DAGSNet (knowledge/architecture.md)
    M        <- zero-shot structured mask: DepGraph groups the layers by their inter- and
                intra-layer dependencies (Fang et al. 2023), the importance of prunable
                dimension k of group g is the L1 group norm (Eq. (7), I(theta) = ||theta||_1),
                and the `sparsity` fraction of the LEAST important channels of every layer
                is removed (uniform per-layer ratio, no data, no gradients)
    theta'   <- M ⊙ theta_0                                                     Eq. (8)
                with the zeroed channels physically removed ("DeepGraph" in the paper),
                so the model is genuinely smaller instead of sparse

The mask is a property of the architecture and the seed, not of any client's data, which
is what lets the server compute it without seeing the data and send it once.

Two representations, one truth:
  * torch-pruning computes the mask and slices a model (`compute_plan`, needs the
    `torch_pruning` package -- only at server initialization);
  * the PLAN it produced -- for every parametrised module, the ORIGINAL indices of the
    channels kept on its output and input dimension -- is recorded as plain lists, and
    `apply_plan` rebuilds the identical pruned module from an unpruned DAGSNet with torch
    alone. `compute_plan` asserts the two agree tensor for tensor before returning, so a
    checkpoint saved with the plan in its cfg rebuilds anywhere.

`sparsity` is torch-pruning's `pruning_ratio`: the fraction of CHANNELS removed per layer
(the paper used Torch-Pruning and calls the same knob "sparsity"). Parameters fall by
roughly (1 - s)^2 for an inner layer since both its input and its output shrink: at
s = 0.7 DAGSNet goes from 395,024 to 35,891 parameters (measured, torch-pruning 1.6.1).
"""
import copy, hashlib, json
import torch
import torch.nn as nn

from proj.model import build_full, n_params

PRUNABLE = (nn.Conv1d, nn.BatchNorm1d, nn.LayerNorm, nn.Linear)
IMPORTANCE = "group_l1"            # Eq. (7): L1 norm per prunable dimension of the group


def _dims(mod):
    """(out_size, in_size) of the two prunable dimensions; in_size None for a module with a
    single feature dimension (BatchNorm, LayerNorm)."""
    if isinstance(mod, nn.Conv1d):
        return mod.out_channels, mod.in_channels
    if isinstance(mod, nn.Linear):
        return mod.out_features, mod.in_features
    if isinstance(mod, nn.BatchNorm1d):
        return mod.num_features, None
    if isinstance(mod, nn.LayerNorm):
        return mod.normalized_shape[0], None
    raise TypeError(type(mod))


def compute_plan(cfg, verbose=False):
    """The server's initialization step. Returns (plan, info): `plan` is None when
    cfg['sparsity'] == 0 (the unpruned baseline), else {module_name: {"out": [...],
    "in": [...] | None}} with the ORIGINAL indices kept; `info` carries the parameter and
    MAC counts before and after, plus the per-layer channel table.

    Deterministic: theta_0 is `torch.manual_seed(cfg['seed']); build_full(cfg)`, exactly
    what proj/driver.py builds as the initial global model, so theta' = M ⊙ theta_0 holds
    channel for channel."""
    s = float(cfg["sparsity"])
    if not 0.0 <= s < 1.0:
        raise ValueError(f"sparsity {s} not in [0, 1)")
    torch.manual_seed(cfg["seed"])
    full = build_full(cfg).eval()
    x = torch.zeros(2, cfg["n_features"])
    info = {"sparsity": s, "importance": IMPORTANCE, "n_params_full": n_params(full)}
    if s == 0.0:
        info.update(n_params=info["n_params_full"], plan_id=None)
        return None, info

    import torch_pruning as tp                     # server-side only
    macs_full, _ = tp.utils.count_ops_and_params(full, x)
    ref = copy.deepcopy(full)                       # theta_0, untouched
    names = {id(m): n for n, m in full.named_modules()}
    # kept[name] = {"out": [original idx still present, in current order], "in": [...]}
    kept = {n: {"out": list(range(_dims(m)[0])),
                "in": list(range(_dims(m)[1])) if _dims(m)[1] is not None else None}
            for n, m in full.named_modules() if isinstance(m, PRUNABLE)}
    imp = tp.importance.GroupMagnitudeImportance(p=1)          # ||.||_1 per channel
    pruner = tp.pruner.BasePruner(full, x, importance=imp, pruning_ratio=s,
                                  ignored_layers=[full.head[5]],  # keep the 16 logits
                                  global_pruning=False)         # uniform ratio per layer
    for group in pruner.step(interactive=True):
        # idxs of every dependency in a group are in the coordinates of the model AS IT IS
        # NOW; collect them, prune, then translate through the current->original maps.
        todo = {}
        for dep, idxs in group:
            mod = dep.target.module
            if not isinstance(mod, PRUNABLE):
                continue                                        # cat / mean / relu / pool
            name = names[id(mod)]
            h = getattr(dep.handler, "__name__", "")
            if "in_channels" in h and kept[name]["in"] is not None:
                dim = "in"
            elif "out_channels" in h or "in_channels" in h:
                dim = "out"                       # BN / LN: one feature dimension
            else:
                raise RuntimeError(f"unknown pruning handler {h!r} on {name}")
            todo.setdefault((name, dim), set()).update(int(i) for i in idxs)
        group.prune()
        for (name, dim), drop in todo.items():
            cur = kept[name][dim]
            if max(drop) >= len(cur):
                raise RuntimeError(f"{name}.{dim}: index {max(drop)} outside {len(cur)}")
            kept[name][dim] = [v for j, v in enumerate(cur) if j not in drop]
    plan = {n: {"out": kept[n]["out"], "in": kept[n]["in"]} for n in kept}
    # ---- the plan must rebuild EXACTLY what torch-pruning built, tensor for tensor
    mine = apply_plan(copy.deepcopy(ref), plan)
    a, b = mine.state_dict(), full.state_dict()
    if list(a) != list(b):
        raise RuntimeError("apply_plan and torch-pruning disagree on the state_dict keys")
    for k in a:
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            raise RuntimeError(f"apply_plan and torch-pruning disagree at {k}: "
                               f"{tuple(a[k].shape)} vs {tuple(b[k].shape)}")
    with torch.no_grad():
        if not torch.equal(mine(x), full(x)):
            raise RuntimeError("rebuilt pruned model and torch-pruning's differ in forward")
    macs_pruned, _ = tp.utils.count_ops_and_params(full, x)
    table = {}
    for n, m in ref.named_modules():
        if isinstance(m, PRUNABLE):
            o, i = _dims(m)
            table[n] = {"type": type(m).__name__, "out": [o, len(plan[n]["out"])],
                        "in": [i, len(plan[n]["in"])] if i is not None else None}
    info.update(n_params=n_params(full), macs_full=int(macs_full), macs_pruned=int(macs_pruned),
                plan_id=plan_id(plan), layers=table,
                torch_pruning=getattr(tp, "__version__", "?"))
    if verbose:
        print(f"[prune] sparsity {s}: {info['n_params_full']:,} -> {info['n_params']:,} params "
              f"({info['n_params'] / info['n_params_full']:.3%}), MACs {macs_full:,} -> "
              f"{macs_pruned:,} ({macs_pruned / macs_full:.3%}), plan {info['plan_id']}")
    return plan, info


def plan_id(plan):
    """Sixteen hex chars over the canonical JSON of the plan: what the fingerprint sees."""
    if plan is None:
        return None
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def apply_plan(model, plan):
    """Slice an UNPRUNED model to the plan, module by module, keeping the listed original
    indices. Every module in the plan must exist with the unpruned dimensions; an index
    outside them or a module the plan does not know is an error, never a silent skip."""
    have = {n for n, m in model.named_modules() if isinstance(m, PRUNABLE)}
    if set(plan) != have:
        raise RuntimeError(f"plan modules {sorted(set(plan) ^ have)} do not match the model")
    for name, sel in plan.items():
        mod = model.get_submodule(name)
        o, i = _dims(mod)
        out = torch.as_tensor(sel["out"], dtype=torch.long)
        inn = torch.as_tensor(sel["in"], dtype=torch.long) if sel["in"] is not None else None
        if out.numel() == 0 or (out.max() >= o) or (inn is not None and (inn.numel() == 0 or inn.max() >= i)):
            raise RuntimeError(f"{name}: plan indices outside the module's dimensions")
        if (inn is None) != (i is None):
            raise RuntimeError(f"{name}: plan has an 'in' list for a single-dimension module")
        dev, dt = mod.weight.device, mod.weight.dtype
        if isinstance(mod, nn.Conv1d):
            new = nn.Conv1d(len(inn), len(out), mod.kernel_size[0], stride=mod.stride[0],
                            padding=mod.padding[0], bias=mod.bias is not None,
                            device=dev, dtype=dt)
            new.weight.copy_(mod.weight[out][:, inn])
            if mod.bias is not None: new.bias.copy_(mod.bias[out])
        elif isinstance(mod, nn.Linear):
            new = nn.Linear(len(inn), len(out), bias=mod.bias is not None, device=dev, dtype=dt)
            new.weight.copy_(mod.weight[out][:, inn])
            if mod.bias is not None: new.bias.copy_(mod.bias[out])
        elif isinstance(mod, nn.BatchNorm1d):
            new = nn.BatchNorm1d(len(out), eps=mod.eps, momentum=mod.momentum,
                                 affine=mod.affine, track_running_stats=mod.track_running_stats,
                                 device=dev, dtype=dt)
            new.weight.copy_(mod.weight[out]); new.bias.copy_(mod.bias[out])
            new.running_mean.copy_(mod.running_mean[out])
            new.running_var.copy_(mod.running_var[out])
            new.num_batches_tracked.copy_(mod.num_batches_tracked)
        elif isinstance(mod, nn.LayerNorm):
            new = nn.LayerNorm(len(out), eps=mod.eps, elementwise_affine=mod.elementwise_affine,
                               device=dev, dtype=dt)
            new.weight.copy_(mod.weight[out]); new.bias.copy_(mod.bias[out])
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)
    return model
