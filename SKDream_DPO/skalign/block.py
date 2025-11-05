import os
import warnings
from typing import Union
from torch import Tensor
from torch import nn
import torch
from functools import partial

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")
    
def _get_norm(indim):
    return nn.LayerNorm(indim)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_linear: bool = True,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_linear = use_linear
        if use_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = partial(torch.cat,dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Union[Tensor,list[Tensor,Tensor,Tensor]], return_attn=False) -> Tensor:
        if self.use_linear:
            B, N, C = x.shape
        else:
            B, N, C = x[0].shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            return attn
        return x


class MemEffAttention(Attention):
    def forward(self, x: Union[Tensor,list[Tensor,Tensor,Tensor]], attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, return_attn)
        if self.use_linear:
            B, N, C = x.shape
            
        else:
            B, N, C = x[0].shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SkAlignBlock(nn.Module):
    def __init__(self,
                 d_model,
                 att_nhead,
                 dim_feedforward=1024,
                 dual_attn=False,
                 droppath=0.0,
                 drop_cross=False,
                 ):
        super().__init__()

        self.d_model = d_model
        self.att_nhead = att_nhead
        self.dual_attn = dual_attn
        self.drop_cross = drop_cross

        # self-attention for x1
        self.norm_x1 = _get_norm(d_model)
        self.self_attn_x1 = MemEffAttention(d_model, att_nhead)

        # feed-forward for boxes
        self.norm_ffn_x1 = _get_norm(d_model)
        self.linear1_x1 = nn.Linear(d_model, dim_feedforward)
        self.activation_x1 = nn.GELU()
        self.linear2_x1 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def forward(self, x_emb):
        
        # self-attention
        _x1_emb = self.norm_x1(x_emb)
        x1_self = self.self_attn_x1(_x1_emb)
        x_emb = x_emb + self.droppath(x1_self)
        
        # Feed-forward
        _x1_emb = self.norm_ffn_x1(x_emb)
        x1_ffn = self.linear2_x1(self.activation_x1(self.linear1_x1(_x1_emb)))
        x_emb = x_emb + self.droppath(x1_ffn)

        return x_emb
    
    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
class SkCrossBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 dual_attn=False,
                 droppath=0.0,
                 drop_cross=False,
                 ):
        super().__init__()

        self.d_model = d_model
        self.att_nhead = att_nhead
        self.dual_attn = dual_attn
        self.drop_cross = drop_cross

        # self-attention for x1
        self.norm_x1 = _get_norm(d_model)
        self.self_attn_x1 = MemEffAttention(d_model, self_nhead)

        # self-attention for x2
        self.norm_x2 = _get_norm(d_model)
        self.self_attn_x2 = MemEffAttention(d_model,self_nhead)
        
        # cross-attention
        self.norm_cross_x2 = _get_norm(d_model)
        self.norm_cross_x1 = _get_norm(d_model)
        self.cross_q_x2 = nn.Linear(d_model,d_model)
        self.cross_kv_x1 = nn.Linear(d_model,2 * d_model)
        self.cross_attn_x2 = MemEffAttention(d_model, att_nhead, use_linear=False)
        if self.dual_attn:
            self.cross_q_x1 = nn.Linear(d_model,d_model)
            self.cross_kv_x2 = nn.Linear(d_model,2 * d_model)
            self.cross_attn_x1 = MemEffAttention(d_model, att_nhead, use_linear=False)

        # feed-forward for x2
        self.norm_ffn_x2 = _get_norm(d_model)
        self.linear1_x2 = nn.Linear(d_model, dim_feedforward)
        self.activation_x2 = nn.GELU()
        self.linear2_x2 = nn.Linear(dim_feedforward, d_model)

        # feed-forward for x1
        self.norm_ffn_x1 = _get_norm(d_model)
        self.linear1_x1 = nn.Linear(d_model, dim_feedforward)
        self.activation_x1 = nn.GELU()
        self.linear2_x1 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def forward(self, x1_emb, x2_emb):
        
        # self-attention for x1
        _x1_emb = self.norm_x1(x1_emb)
        x1_self = self.self_attn_x1(_x1_emb)
        x1_emb = x1_emb + self.droppath(x1_self)
        
        # self-attention for x2
        _x2_emb = self.norm_x2(x2_emb)
        x2_self = self.self_attn_x2(_x2_emb)
        x2_emb = x2_emb + self.droppath(x2_self)

        # cross atention
        _x1_emb = self.norm_cross_x1(x1_emb)
        _x2_emb = self.norm_cross_x2(x2_emb)

        q = self.cross_q_x2(_x2_emb)
        kv = self.cross_kv_x1(_x1_emb)
        k,v = torch.split(kv,self.d_model,dim=-1)
        x1_cross = self.cross_attn_x1([q,k,v])
        if self.drop_cross:
            x1_emb = x1_emb + self.droppath(x1_cross)
        else:
            x1_emb = x1_emb + x1_cross

        if self.dual_attn:
            q = self.cross_q_x1(_x1_emb)
            kv = self.cross_kv_x2(_x2_emb)
            k,v = torch.split(kv,self.d_model,dim=-1)
            x2_cross = self.cross_attn_x2([q,k,v])
            if self.drop_cross:
                x2_emb = x2_emb + self.droppath(x2_cross)
            else:
                x2_emb = x2_emb + x2_cross

        # Feed-forward
        _x1_emb = self.norm_ffn_x1(x1_emb)
        x1_ffn = self.linear2_x1(self.activation_x1(self.linear1_x1(_x1_emb)))
        x1_emb = x1_emb + self.droppath(x1_ffn)
        
        _x2_emb = self.norm_ffn_x2(x2_emb)
        x2_ffn = self.linear2_x2(self.activation_x2(self.linear1_x2(_x2_emb)))
        x2_emb = x2_emb + self.droppath(x2_ffn)

        

        return x1_emb,x2_emb
    
    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)