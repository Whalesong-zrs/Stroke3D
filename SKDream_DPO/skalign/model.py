import torch.nn as nn
import torch
import math
from skalign.block import SkAlignBlock
class SkalignModel(nn.Module):
    def __init__(self, d_in, num_block = 1) -> None:
        super().__init__()

        blocks = []
        for i in range(num_block):
            blocks.append(SkAlignBlock(d_in,d_in//64,d_in//64,dual_attn=True))
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self,x):
        B,N,C = x.shape
        S = int(math.sqrt(N))
        for block in self.blocks:
            x = block(x)
        x = torch.mean(x,dim=1)
        # return x.view(B,N*C)
        return x


