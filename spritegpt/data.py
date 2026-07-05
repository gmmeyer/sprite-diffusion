"""Dataset: the whole packed npz lives on the GPU; batches are random indexes.

At 32-64px and <100k images this is far faster than a DataLoader.
"""

import numpy as np
import torch


class GpuImageDataset:
    def __init__(
        self,
        npz_path: str,
        device: str = "cuda",
        hflip: bool = True,
        text_npz: str | None = None,
    ):
        d = np.load(npz_path, allow_pickle=False)
        self.images = torch.from_numpy(d["images"]).to(device)  # [N,H,W,3] uint8
        self.group_ids = torch.from_numpy(d["group_ids"]).to(device)
        self.group_names = [str(g) for g in d["group_names"]]
        self.num_classes = len(self.group_names)
        self.hflip = hflip
        self.device = device
        self.ctx = None
        if text_npz is not None:
            t = np.load(text_npz, allow_pickle=False)
            assert len(t["ctx"]) == len(self.images), "text npz misaligned with images"
            self.ctx = torch.from_numpy(t["ctx"]).to(device)  # [N,L,512] fp16
            self.ctx_mask = torch.from_numpy(t["ctx_mask"]).to(device)
            self.null_ctx = torch.from_numpy(t["null_ctx"]).to(device)
            self.null_mask = torch.from_numpy(t["null_mask"]).to(device)

    def __len__(self) -> int:
        return self.images.shape[0]

    def _images_at(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.images[idx].permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if self.hflip:
            flip = torch.rand(len(idx), device=self.device) < 0.5
            x[flip] = x[flip].flip(-1)
        return x

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (images in [-1,1] NCHW, group ids)."""
        idx = torch.randint(len(self), (batch_size,), device=self.device)
        return self._images_at(idx), self.group_ids[idx]

    def batch_text(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (images in [-1,1] NCHW, caption ctx, ctx mask)."""
        idx = torch.randint(len(self), (batch_size,), device=self.device)
        return self._images_at(idx), self.ctx[idx], self.ctx_mask[idx]
