"""Rectified flow (flow matching) objective and Euler ODE sampler.

Convention: t=0 is data, t=1 is noise.
  x_t = (1 - t) * x0 + t * eps
  target velocity v = eps - x0, so integrating dx/dt = v from t=1 to t=0
  with x_1 ~ N(0, I) yields data.

Conditioning is passed through to the model: class ids (y) and/or CLIP text
token embeddings (ctx, ctx_mask). For classifier-free guidance the sampler
needs the unconditional stand-in: the null class index for y, or the
empty-string embedding (null_ctx, null_mask) for text.
"""

import torch
import torch.nn.functional as F
from torch import nn


def rf_loss(
    model: nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor | None = None,
    ctx: torch.Tensor | None = None,
    ctx_mask: torch.Tensor | None = None,
    time_sampling: str = "logit_normal",
) -> torch.Tensor:
    b = x0.shape[0]
    if time_sampling == "logit_normal":  # SD3-style, weights mid timesteps
        t = torch.sigmoid(torch.randn(b, device=x0.device))
    else:
        t = torch.rand(b, device=x0.device)
    eps = torch.randn_like(x0)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * eps
    v = model(xt, t, y, ctx, ctx_mask)
    return F.mse_loss(v, eps - x0)


@torch.no_grad()
def sample(
    model: nn.Module,
    n: int,
    img_size: int,
    steps: int = 50,
    y: torch.Tensor | None = None,
    ctx: torch.Tensor | None = None,
    ctx_mask: torch.Tensor | None = None,
    null_ctx: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    guidance: float = 0.0,
    num_classes: int | None = None,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler sampler. Returns images in [-1, 1]."""
    cond = y is not None or ctx is not None
    x = torch.randn(n, 3, img_size, img_size, device=device, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n)
        v = model(x, t, y, ctx, ctx_mask)
        if guidance > 0 and cond:
            null_y = None
            if y is not None:
                null_y = torch.full((n,), num_classes, device=device, dtype=torch.long)
            u_ctx = null_ctx.expand(n, -1, -1) if ctx is not None else None
            u_mask = null_mask.expand(n, -1) if ctx is not None else None
            v_uncond = model(x, t, null_y, u_ctx, u_mask)
            v = v_uncond + guidance * (v - v_uncond)
        x = x - (ts[i] - ts[i + 1]) * v
    return x.clamp(-1, 1)
