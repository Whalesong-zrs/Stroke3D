import torch
import torch.nn as nn

from skalign.block import SkAlignBlock


class SkalignModel(nn.Module):
    def __init__(self, d_in, num_block=1) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [SkAlignBlock(d_in, d_in // 64, d_in // 64, dual_attn=True) for _ in range(num_block)]
        )
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return torch.mean(x, dim=1)

