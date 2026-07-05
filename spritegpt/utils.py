import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_image_grid(x: torch.Tensor, path: str | Path, nrow: int | None = None) -> None:
    """x: [N,3,H,W] in [-1,1] -> PNG grid."""
    n, _, h, w = x.shape
    nrow = nrow or math.ceil(math.sqrt(n))
    ncol = math.ceil(n / nrow)
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    grid = np.full((ncol * h, nrow * w, 3), 255, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = arr[i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
