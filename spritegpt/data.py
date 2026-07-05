"""Dataset: the whole packed npz lives on the GPU; batches are random indexes.

At 32-64px and <100k images this is far faster than a DataLoader.
"""

import numpy as np
import torch


class GpuImageDataset:
    def __init__(self, npz_path: str, device: str = "cuda", hflip: bool = True):
        d = np.load(npz_path, allow_pickle=False)
        self.images = torch.from_numpy(d["images"]).to(device)  # [N,H,W,3] uint8
        self.group_ids = torch.from_numpy(d["group_ids"]).to(device)
        self.group_names = [str(g) for g in d["group_names"]]
        self.num_classes = len(self.group_names)
        self.hflip = hflip
        self.device = device

    def __len__(self) -> int:
        return self.images.shape[0]

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (images in [-1,1] NCHW, group ids)."""
        idx = torch.randint(len(self), (batch_size,), device=self.device)
        x = self.images[idx].permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if self.hflip:
            flip = torch.rand(batch_size, device=self.device) < 0.5
            x[flip] = x[flip].flip(-1)
        return x, self.group_ids[idx]
