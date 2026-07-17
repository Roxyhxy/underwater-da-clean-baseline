import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationGatedLoRALinear(nn.Module):
    """Frozen linear layer with a zero-initialized low-rank residual.

    In ``gated`` mode, a per-image degradation descriptor controls the LoRA
    rank channels. In ``plain`` mode, the same LoRA path is used without
    image-dependent conditioning.
    """

    def __init__(
        self,
        linear,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        condition_dim=128,
        mode="gated",
    ):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("DegradationGatedLoRALinear expects nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if mode not in {"plain", "gated"}:
            raise ValueError("LoRA mode must be 'plain' or 'gated'")

        # Keep the original parameter names (weight/bias), so a stock DA2
        # checkpoint remains directly loadable after LoRA injection.
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.mode = mode
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, linear.out_features, bias=False)
        self.condition_gate = nn.Linear(int(condition_dim), self.rank) if mode == "gated" else None
        self.enabled = True
        self._condition = None

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        if self.condition_gate is not None:
            nn.init.zeros_(self.condition_gate.weight)
            nn.init.zeros_(self.condition_gate.bias)

    def set_condition(self, condition):
        self._condition = condition

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return base

        residual = self.lora_a(self.dropout(x))
        if self.mode == "gated" and self._condition is not None:
            condition = self._condition.to(device=x.device, dtype=x.dtype)
            gate = 2.0 * torch.sigmoid(self.condition_gate(condition))
            residual = residual * gate[:, None, :]
        return base + self.lora_b(residual) * self.scaling


def inject_qkv_lora(
    backbone,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    condition_dim=128,
    mode="gated",
    last_n_blocks=12,
):
    """Replace QKV linears in the selected last DINO blocks with LoRA."""
    n_blocks = len(backbone.blocks)
    first_block = max(0, n_blocks - int(last_n_blocks))
    injected = []

    for block_index, block in enumerate(backbone.blocks):
        if block_index < first_block:
            continue
        qkv = block.attn.qkv
        if isinstance(qkv, DegradationGatedLoRALinear):
            raise RuntimeError("QKV LoRA has already been injected")
        block.attn.qkv = DegradationGatedLoRALinear(
            qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            condition_dim=condition_dim,
            mode=mode,
        )
        injected.append(f"blocks.{block_index}.attn.qkv")

    if not injected:
        raise RuntimeError("No DINO QKV layers were selected for LoRA injection")
    return injected
