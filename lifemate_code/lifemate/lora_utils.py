"""Pure-PyTorch LoRA Engine (Zero dependency on `peft` — immune to `transformers 5.x` / Python 3.13 dispatcher bugs).

Provides:
- `PureLoRALinear`: Drop-in `nn.Module` replacement for `nn.Linear` with `float32` master LoRA parameters
  and native TensorCore `bf16`/`fp16` forward/backward passes.
- `inject_pure_lora`: Replaces target linear modules in-place on any `PreTrainedModel` (`Qwen3ForCausalLM`).
- `save_pure_lora`: Saves lightweight (~65 MB) `pure_lora_weights.pt` + `pure_lora_config.json`.
- `load_and_merge_pure_lora`: Merges trained LoRA deltas directly into base model weights in <0.1s at inference.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class PureLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        dev = base_layer.weight.device
        # Master LoRA weights in float32: works seamlessly with both bf16 and fp16 GradScaler + high-precision AdamW
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_d = self.dropout(x)
        a = self.lora_A.to(dtype=x_d.dtype)
        b = self.lora_B.to(dtype=x_d.dtype)
        lora_out = F.linear(F.linear(x_d, a), b)
        return base_out + (lora_out * self.scaling).to(dtype=base_out.dtype)


def inject_pure_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] | list[str] = DEFAULT_TARGET_MODULES,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> int:
    """Freeze all base parameters and replace matching `nn.Linear` layers with `PureLoRALinear`."""
    for p in model.parameters():
        p.requires_grad_(False)

    target_set = set(target_modules)
    to_replace: list[tuple[nn.Module, str, nn.Linear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.rsplit(".", 1)[-1]
        if leaf_name in target_set:
            if "." in full_name:
                parent_path, child_name = full_name.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
            else:
                parent, child_name = model, full_name
            to_replace.append((parent, child_name, module))

    for parent, child_name, linear_mod in to_replace:
        setattr(parent, child_name, PureLoRALinear(linear_mod, r=r, alpha=alpha, dropout=dropout))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    print(f"[PureLoRA] Injected {len(to_replace)} linear layers | trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return len(to_replace)


def save_pure_lora(
    model: nn.Module,
    out_dir: Path,
    base_model_name: str,
    r: int,
    alpha: int,
    target_modules: list[str] | tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for full_name, module in model.named_modules():
        if isinstance(module, PureLoRALinear):
            weights[full_name] = {
                "lora_A": module.lora_A.detach().cpu().to(torch.float16),
                "lora_B": module.lora_B.detach().cpu().to(torch.float16),
            }

    cfg = {
        "lora_format": "pure_pytorch_v1",
        "base_model_name_or_path": base_model_name,
        "r": r,
        "lora_alpha": alpha,
        "target_modules": list(target_modules),
        "num_Linear_modules": len(weights),
    }
    with open(out_dir / "pure_lora_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Also write adapter_config.json marker so existing notebook checks (`os.path.isfile(.../adapter_config.json)`) detect it automatically
    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    torch.save(weights, out_dir / "pure_lora_weights.pt")


def load_and_merge_pure_lora(base_model: nn.Module, adapter_dir: Path) -> nn.Module:
    cfg = json.load(open(adapter_dir / "pure_lora_config.json", encoding="utf-8"))
    scaling = float(cfg["lora_alpha"]) / float(cfg["r"])
    weights: dict[str, dict[str, torch.Tensor]] = torch.load(
        adapter_dir / "pure_lora_weights.pt", map_location="cpu", weights_only=True
    )

    merged_count = 0
    with torch.no_grad():
        for full_name, ab in weights.items():
            mod = base_model.get_submodule(full_name)
            w = mod.weight
            a = ab["lora_A"].to(device=w.device, dtype=torch.float32)
            b = ab["lora_B"].to(device=w.device, dtype=torch.float32)
            delta = (b @ a) * scaling
            w.data.add_(delta.to(dtype=w.dtype))
            merged_count += 1

    print(f"[PureLoRA] Merged {merged_count} LoRA layers from {adapter_dir} into base model!")
    return base_model


def load_pure_lora_state(adapter_dir: Path, device: str = "cuda") -> dict:
    """Load a PureLoRA adapter into memory for fast step-level multi-adapter switching (`MultiLoRAEnsembleBackend`)."""
    adapter_dir = Path(adapter_dir)
    cfg = json.load(open(adapter_dir / "pure_lora_config.json", encoding="utf-8"))
    weights: dict[str, dict[str, torch.Tensor]] = torch.load(
        adapter_dir / "pure_lora_weights.pt", map_location=device, weights_only=True
    )
    return {
        "name": adapter_dir.name,
        "path": str(adapter_dir),
        "r": int(cfg["r"]),
        "alpha": int(cfg["lora_alpha"]),
        "scaling": float(cfg["lora_alpha"]) / float(cfg["r"]),
        "target_modules": tuple(cfg.get("target_modules", DEFAULT_TARGET_MODULES)),
        "weights": weights,
    }


def apply_pure_lora_state(model: nn.Module, state: dict) -> None:
    """Swap active PureLoRA weights in-place (< 2 ms on GPU) without duplicating the 4B base model."""
    weights = state["weights"]
    scaling = state["scaling"]
    with torch.no_grad():
        for full_name, module in model.named_modules():
            if isinstance(module, PureLoRALinear) and full_name in weights:
                ab = weights[full_name]
                module.scaling = scaling
                module.lora_A.data.copy_(ab["lora_A"].to(device=module.lora_A.device, dtype=module.lora_A.dtype))
                module.lora_B.data.copy_(ab["lora_B"].to(device=module.lora_B.device, dtype=module.lora_B.dtype))

