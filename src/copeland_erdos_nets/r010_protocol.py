"""R010 paired attention-init protocol primitives.

Separate from the historical screening runner. Binding DS amendments
from R010_DESIGN_DS_REVIEW.md (2026-08-24).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from copeland_erdos_nets.assignment import compute_effective_rank
from copeland_erdos_nets.ce_init import ce_init_

ALLOWLIST_SUFFIXES = (
    ".attn.q_proj.weight",
    ".attn.k_proj.weight",
    ".attn.v_proj.weight",
    ".attn.out_proj.weight",
)

METHODS = (
    "xavier_g1.0",
    "orthogonal",
    "xavier_g1.2",
    "ce_lcg",
)

PRIMARY_BASELINE = "xavier_g1.0"

METHOD_ATTN_SALT = {
    "xavier_g1.0": 0,
    "orthogonal": 1,
    "xavier_g1.2": 2,
    "ce_lcg": 3,
}

SEED_MODEL_OFFSET = 0
SEED_ATTENTION_OFFSET = 10007
SEED_SHUFFLE_OFFSET = 20011
ATTN_STREAM_STRIDE = 1_000_003


@dataclass(frozen=True)
class R010Seeds:
    experiment: int
    seed_model: int
    seed_attention: int
    seed_shuffle: int

    def attention_seed_for(self, method: str) -> int:
        if method not in METHOD_ATTN_SALT:
            raise ValueError(f"unknown method {method!r}")
        return self.seed_attention + METHOD_ATTN_SALT[method] * ATTN_STREAM_STRIDE


def derive_seeds(experiment_seed: int) -> R010Seeds:
    return R010Seeds(
        experiment=int(experiment_seed),
        seed_model=int(experiment_seed) + SEED_MODEL_OFFSET,
        seed_attention=int(experiment_seed) + SEED_ATTENTION_OFFSET,
        seed_shuffle=int(experiment_seed) + SEED_SHUFFLE_OFFSET,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def hash_int_sequence(values: Iterable[int]) -> str:
    payload = ",".join(str(int(v)) for v in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def collect_named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        named[name] = param
    for name, buf in model.named_buffers():
        named[f"buffer:{name}"] = buf
    return named


def attention_allowlist(model: nn.Module) -> list[str]:
    names = [
        name
        for name, _ in model.named_parameters()
        if name.endswith(ALLOWLIST_SUFFIXES)
    ]
    names.sort()
    return names


def assert_expected_allowlist_count(allowlist: list[str], n_layers: int) -> None:
    expected = n_layers * 4
    if len(allowlist) != expected:
        raise AssertionError(
            f"allowlist size {len(allowlist)} != {expected} (2-layer expect 8)"
        )
    for name in allowlist:
        if not name.endswith(ALLOWLIST_SUFFIXES):
            raise AssertionError(f"non-allowlisted name leaked: {name}")


def _is_non_attention_linear(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return ".attn." not in f"{module_name}."


def apply_historical_non_attention_xavier(model: nn.Module) -> None:
    """Xavier on non-attention Linear weights; zero those biases.

    Attention Linear tensors stay at constructor values until the shared
    intervention API overwrites allowlisted weights (including baseline).
    """
    for name, module in model.named_modules():
        if not _is_non_attention_linear(name, module):
            continue
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def zero_all_linear_biases(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


def build_base_state(
    factory: Callable[[], nn.Module],
    seed_model: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, str]]:
    torch.manual_seed(seed_model)
    np.random.seed(seed_model % (2**32 - 1))
    model = factory()
    apply_historical_non_attention_xavier(model)
    zero_all_linear_biases(model)
    model.to(device)
    hashes = {
        name: tensor_sha256(tensor)
        for name, tensor in collect_named_tensors(model).items()
    }
    return model, hashes


def clone_from_base_state(base_model: nn.Module, factory: Callable[[], nn.Module]) -> nn.Module:
    clone = factory()
    clone.load_state_dict(base_model.state_dict())
    return clone


def _cpu_init_with_seed(weight: torch.Tensor, seed: int, fn) -> None:
    device = weight.device
    cpu_w = torch.empty_like(weight, device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) % (2**31 - 1))
        fn(cpu_w)
    weight.copy_(cpu_w.to(device))


def apply_attention_intervention(
    model: nn.Module,
    method: str,
    seeds: R010Seeds,
    *,
    allowlist: list[str] | None = None,
    ce_m: int = 8,
    ce_offset_blocks: int = 0,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unauthorized method {method!r}")
    allowlist = list(allowlist if allowlist is not None else attention_allowlist(model))
    params = dict(model.named_parameters())
    attn_seed = seeds.attention_seed_for(method)

    for i, name in enumerate(allowlist):
        weight = params[name]
        tensor_seed = attn_seed + 1 + i
        if method == "xavier_g1.0":
            _cpu_init_with_seed(
                weight.data, tensor_seed, lambda w: nn.init.xavier_normal_(w, gain=1.0)
            )
        elif method == "xavier_g1.2":
            _cpu_init_with_seed(
                weight.data, tensor_seed, lambda w: nn.init.xavier_normal_(w, gain=1.2)
            )
        elif method == "orthogonal":
            _cpu_init_with_seed(weight.data, tensor_seed, nn.init.orthogonal_)
        elif method == "ce_lcg":
            ce_init_(
                weight.data,
                m=ce_m,
                offset_blocks=ce_offset_blocks,
                kind="he",
                mode="normal",
                assignment="lcg",
            )

    return {
        "method": method,
        "seed_model": seeds.seed_model,
        "seed_attention": seeds.seed_attention,
        "seed_attention_method": attn_seed,
        "seed_shuffle": seeds.seed_shuffle,
        "ce_m": ce_m,
        "ce_offset_blocks": ce_offset_blocks,
    }


def assert_t0_invariance(
    model: nn.Module,
    base_hashes: dict[str, str],
    allowlist: list[str],
) -> dict[str, str]:
    current = collect_named_tensors(model)
    changed: list[str] = []
    unchanged: dict[str, str] = {}
    allow = set(allowlist)
    for name, tensor in current.items():
        digest = tensor_sha256(tensor)
        base = base_hashes.get(name)
        if base is None:
            raise AssertionError(f"tensor {name} missing from base_state")
        if name in allow:
            if digest == base:
                unchanged[name] = digest
            else:
                changed.append(name)
            continue
        if digest != base:
            raise AssertionError(
                f"invariance failure: {name} changed vs base_state "
                f"(was {base[:12]} now {digest[:12]})"
            )
        unchanged[name] = digest
    if not changed:
        raise AssertionError("allowlisted attention weights did not change vs base_state")
    missing = allow - set(changed) - set(unchanged)
    if missing:
        raise AssertionError(f"allowlist tensors missing after intervention: {sorted(missing)}")
    return unchanged


def spectral_row(name: str, weight: torch.Tensor, *, state: str, method: str, seed: int) -> dict:
    w = weight.detach().float().cpu()
    matrix = w.numpy()
    if matrix.ndim != 2:
        raise ValueError(f"{name} is not 2-D")
    s = np.linalg.svd(matrix, compute_uv=False)
    cond = float(s[0] / max(s[-1], 1e-12))
    erank = float(compute_effective_rank(w))
    return {
        "state": state,
        "method": method,
        "seed": seed,
        "name": name,
        "std": float(w.std().item()),
        "norm": float(w.norm().item()),
        "condition_number": cond,
        "effective_rank": erank,
        "sv_max": float(s[0]),
        "sv_median": float(np.median(s)),
        "sv_min": float(s[-1]),
        "sv_mean": float(s.mean()),
    }


def collect_spectral(
    model: nn.Module,
    allowlist: list[str],
    *,
    state: str,
    method: str,
    seed: int,
) -> list[dict]:
    params = dict(model.named_parameters())
    return [
        spectral_row(name, params[name], state=state, method=method, seed=seed)
        for name in allowlist
    ]


def epoch_index_permutations(
    n_items: int,
    n_epochs: int,
    seed_shuffle: int,
) -> list[list[int]]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_shuffle)
    return [torch.randperm(n_items, generator=g).tolist() for _ in range(n_epochs)]


def batch_order_records(
    perms: list[list[int]],
    *,
    seed: int,
    batch_size: int,
    drop_last: bool,
) -> list[dict]:
    rows = []
    for epoch, order in enumerate(perms, start=1):
        usable = order
        if drop_last:
            n = (len(order) // batch_size) * batch_size
            usable = order[:n]
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "n_indices": len(usable),
                "batch_order_hash": hash_int_sequence(usable),
            }
        )
    return rows


def dump_json(path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
