import torch
from torch import Tensor

class DINOProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Conv2d(1536, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, imgs: Tensor) -> Tensor:
        # imgs has shape (batch_size, n_views, c, h, w)
        batch_size, n_views, c, h, w = imgs.shape
        # convert it to (batch_size * n_views, c, h, w) to give to Conv2d
        flat_imgs = imgs.flatten(0, 1)
        feats = self.proj(flat_imgs)
        _, c_proj, h_proj, w_proj = feats.shape
        unflattened_feats = feats.reshape(batch_size, n_views, c_proj, h_proj, w_proj)
        return unflattened_feats
