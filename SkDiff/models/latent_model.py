# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from timm.models.vision_transformer import Mlp
import math
from typing import Optional, List

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        self.output_dims = self.input_dims * self.num_freqs * 2
        if self.include_input:
            self.output_dims += self.input_dims
            
        freq_bands = 2.0 ** torch.linspace(0.0, self.num_freqs - 1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        p_times_freq = x.unsqueeze(-1) * self.freq_bands

        p_sin = torch.sin(p_times_freq * math.pi)
        p_cos = torch.cos(p_times_freq * math.pi)

        encoding = torch.cat([p_sin, p_cos], dim=-1)

        encoding = encoding.view(*x.shape[:-1], -1)

        if self.include_input:
            encoding = torch.cat([x, encoding], dim=-1)
            
        return encoding
    

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_ratio=4.0,):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        
        # reference Wan
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, edge_index, t, context):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        #self attention
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), edge_index)
        # cross attn
        query_mha = self.norm3(x).unsqueeze(1) # N, 1, hidden_dim
        keys_values_for_mha = context
        cross_attn_output, _ = self.cross_attn(
            query=query_mha,
            key=keys_values_for_mha,
            value=keys_values_for_mha
        )
        cross_attn_output = cross_attn_output.squeeze(1)
        x = x + cross_attn_output

        # ffn
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x 


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class TextLatentModel(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 depth: int,
                 clip_embed_dim: int,
                 heads: int,
                 dropout: float,
                 input_dims=2, 
                 num_freqs=10,
                 include_input=True,
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.clip_embed_dim = clip_embed_dim

        self.x_embedder = nn.Linear(latent_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.pos_encoder = PositionalEncoding(input_dims=input_dims, num_freqs=num_freqs)
        if include_input:
            frequency_embedding_size = 2*input_dims*num_freqs + input_dims
        else:
            frequency_embedding_size = 2*input_dims*num_freqs

        self.pos_embedder = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.context_embedder = nn.Linear(self.clip_embed_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(TransformerBlock(hidden_dim=self.hidden_dim, heads=heads))

        self.final_layer = FinalLayer(hidden_dim=self.hidden_dim, output_dim=latent_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        torch.nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                t: torch.Tensor,
                batch_node,
                node_xy: torch.Tensor,
                context: torch.Tensor,
                controlnet_residuals: Optional[List[torch.Tensor]] = None
               ):

        x = self.x_embedder(x)

        pos_encoding = self.pos_encoder(node_xy)
        pos_emb = self.pos_embedder(pos_encoding)
        x = x + pos_emb

        t = t.float()
        t = self.t_embedder(t)

        assert context is not None 
        context = context.to(x.device)
        context = self.context_embedder(context)
        context = context[batch_node]
        
        for i in range(self.depth):
            x = self.blocks[i](x, edge_index, t, context)

            if controlnet_residuals is not None:
                if i < len(controlnet_residuals):
                    x = x + controlnet_residuals[i]

        x = self.final_layer(x, t)
        return x