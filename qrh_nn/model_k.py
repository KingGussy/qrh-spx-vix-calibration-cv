from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from qrh_nn.model import _make_activation

''' Same model archtecture, but here k is fed as an input feature so that the resulting model is cts in k'''

# ============================================================
# Helpers
# ============================================================

# def _make_activation(name: str) -> nn.Module:
#     name = name.lower()
#     if name == "relu":
#         return nn.ReLU()
#     if name == "gelu":
#         return nn.GELU()
#     if name == "silu" or name == "swish":
#         return nn.SiLU()
#     raise ValueError(f"Unknown activation: {name}")


# ============================================================
# Config
# ============================================================

@dataclass
class ContinuousKConfig:
    # Input is (omega, z0, T, k) = 17 dims when u is dropped
    d_in: int = 17
    d_out: int = 1

    d_model: int = 256
    d_hidden: int = 512
    n_blocks: int = 6

    dropout: float = 0.0
    use_layernorm: bool = True
    act: str = "silu"
    out_act: str | None = None


# ============================================================
# Residual block
# ============================================================

class ContinuousKBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        act: str = "silu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = _make_activation(act)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)

        # ReZero-style residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + self.alpha * y


# ============================================================
# Model A: scratch continuous-k model
# ============================================================

class ContinuousKModel(nn.Module):
    """
    Model A:
      input  = (omega, z0, T, k)    [17 dims]
      output = sigma_spx            [1 dim]
    """
    def __init__(self, cfg: ContinuousKConfig):
        super().__init__()
        self.cfg = cfg

        self.inp = nn.Linear(cfg.d_in, cfg.d_model)

        self.blocks = nn.Sequential(
            *[
                ContinuousKBlock(
                    d_model=cfg.d_model,
                    d_hidden=cfg.d_hidden,
                    act=cfg.act,
                    dropout=cfg.dropout,
                    use_layernorm=cfg.use_layernorm,
                )
                for _ in range(cfg.n_blocks)
            ]
        )

        self.out = nn.Linear(cfg.d_model, cfg.d_out)
        self.out_act = _make_activation(cfg.out_act) if cfg.out_act else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.inp.weight, a=math.sqrt(5))
        nn.init.zeros_(self.inp.bias)

        nn.init.kaiming_uniform_(self.out.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 17)
        returns: (B, 1)
        """
        h = self.inp(x)
        h = self.blocks(h)
        y = self.out(h)
        y = self.out_act(y)
        return y



def build_ctsk_model(
    d_in: int = 17,
    d_model: int = 256,
    d_hidden: int = 512,
    n_blocks: int = 6,
    dropout: float = 0.0,
    act: str = "silu",
) -> ContinuousKModel:
    cfg = ContinuousKConfig(
        d_in=d_in,
        d_model=d_model,
        d_hidden=d_hidden,
        n_blocks=n_blocks,
        dropout=dropout,
        act=act,
    )
    return ContinuousKModel(cfg)


if __name__ == "__main__":
    model = build_ctsk_model()
    x = torch.randn(8, 17)
    y = model(x)
    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))