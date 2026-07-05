"""Exponential moving average of model parameters, kept in fp32."""

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.step = 0
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        # warmup: effective decay ramps up from 0 so early EMA isn't stuck at init
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.detach().float(), 1 - decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.step = state["step"]
        self.shadow = state["shadow"]
