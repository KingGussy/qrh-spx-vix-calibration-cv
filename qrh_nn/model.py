from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

''' This is our fixed-k model, ResMLP'''
@dataclass
class ResMLPConfig:
    d_in:  int = 16
    d_out: int = 30
    d_model:  int = 256 
    d_hidden: int = 512 # hidden size each resid block
    n_blocks:  int = 6
    dropout:  float = 0.0
    use_layernorm: bool = True,
    act: str = "silu",
    out_act: Optional[str] = None

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GeLU()
    raise ValueError(f"Unknown activation fucntion '{name}' ")

def _make_out_activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name == "softplus":
        return nn.Softplus()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown out activation '{name}'")

class ResBlock(nn.Module):
    """
    One block is Linea -> Activation (dropout) -> linear
    """
    def __init__(
            self,
            d_model:  int, 
            d_hidden: int,
            dropout:  float = 0.0,
            act:      str = "silu",
            use_layernorm: bool = True,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = _make_activation(act)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(d_hidden, d_model)

        # ReZero scaling
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        # conservative init for stability
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h) # unused rn
        h = self.fc2(h)
        return x + h     # not using alpha rn
    
class ResMLP(nn.Module):
    """
    Full ResMLP for MtP approach: (params, z0, T) --> (SPX_IV, VIX_IV)
    
    Forward:
        x -> Linear(d_in -> d_model) --> activation --> ResBlock x6 --> Linear(d_model, d_out)
    """
    def __init__(self, cfg: ResMLPConfig):
        super().__init__()
        self.cfg = cfg

        self.inp = nn.Linear(cfg.d_in, cfg.d_model)
        self.act = _make_activation(cfg.act)

        self.blocks = nn.Sequential(
            *[ ResBlock(
                d_model=cfg.d_model, 
                d_hidden=cfg.d_hidden,
                dropout=cfg.dropout,
                act=cfg.act,
                use_layernorm=cfg.use_layernorm
            )
            for _ in range(cfg.n_blocks)
            ]
        )

        self.out = nn.Linear(cfg.d_model, cfg.d_out)
        self.out_act = _make_out_activation(cfg.out_act)
        
        self._init_weights()

    def _init_weights(self):
        # input/output layer- kaiming init
        nn.init.kaiming_uniform_(self.inp.weight, a=math.sqrt(5))
        nn.init.zeros_(self.inp.bias)

        nn.init.kaiming_uniform_(self.out.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.out(x)
        x = self.out_act(x)
        return x
        
def build_resmlp(
    d_in: int = 16,
    d_out: int = 30,
    d_model: int = 256,
    d_hidden: int = 512,
    n_blocks: int = 6,
    dropout: float = 0.0,
    use_layernorm: bool = True,
    act: str = "silu",
) -> ResMLP:
    cfg = ResMLPConfig(
        d_in=d_in,
        d_out=d_out,
        d_model=d_model,
        d_hidden=d_hidden,
        n_blocks=n_blocks,
        dropout=dropout,
        use_layernorm=use_layernorm,
        act=act,
        out_act=None,
    )
    return ResMLP(cfg)
