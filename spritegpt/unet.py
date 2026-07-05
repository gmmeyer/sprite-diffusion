"""Pixel-space U-Net for small-image diffusion (ADM-style, simplified).

Scale-shift GroupNorm conditioning on (timestep + optional class) embedding,
self-attention at the resolutions given in attn_res.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding. t is in [0, 1]; scaled by 1000 for frequency coverage."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2)  # scale, shift
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch: int, head_dim: int = 64):
        super().__init__()
        self.num_heads = max(1, ch // head_dim)
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hgt, wid = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, hgt * wid).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(b, c, hgt, wid)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        in_ch: int = 3,
        base: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 2),
        num_res: int = 2,
        attn_res: tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.base = base
        if num_classes is not None:
            # index num_classes = null token for classifier-free guidance
            self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        chans = [base]  # channel count of each skip tensor
        ch, res = base, img_size
        self.down = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res):
                block = nn.ModuleList([ResBlock(ch, base * mult, emb_dim, dropout)])
                ch = base * mult
                if res in attn_res:
                    block.append(AttnBlock(ch))
                self.down.append(block)
                chans.append(ch)
            if level < len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList(
            [ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, emb_dim, dropout)]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            for i in range(num_res + 1):
                block = nn.ModuleList([ResBlock(ch + chans.pop(), base * mult, emb_dim, dropout)])
                ch = base * mult
                if res in attn_res:
                    block.append(AttnBlock(ch))
                if level > 0 and i == num_res:
                    block.append(Upsample(ch))
                    res *= 2
                self.up.append(block)

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = zero_init(nn.Conv2d(ch, in_ch, 3, padding=1))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base))
        if self.num_classes is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)

        h = self.conv_in(x)
        skips = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)

        for block in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))
