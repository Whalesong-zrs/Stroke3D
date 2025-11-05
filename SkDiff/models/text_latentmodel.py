import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from timm.models.vision_transformer import Mlp
import math
from typing import Optional, List

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


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

    def forward(self, x, edge_index, t, text_emb):
        # return
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        # self attn
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), edge_index)

        # cross attn
        query_mha = self.norm3(x).unsqueeze(1) # N, 1, hidden_dim
        keys_values_for_mha = text_emb
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
    

class LatentModel(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 depth: int,
                 clip_token_embed_dim: int,
                 dropout: float,
                 heads: int,
                cross_attn_dropout: float = 0
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        self.x_embedder = nn.Linear(latent_dim, hidden_dim)
        self.text_embedder = nn.Linear(clip_token_embed_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

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
                batch_node: torch.Tensor,
                num_graphs: int,              # (可选) 批处理中的图数量, 可从 caption_emb.shape[0] 或 batch_node 推断
                caption_emb: Optional[torch.Tensor] = None, # CLIP token嵌入序列 [num_graphs, vocab_len, clip_token_embed_dim]
                controlnet_residuals: Optional[List[torch.Tensor]] = None
               ):
        x = self.x_embedder(x)
        projected_text_emb = self.text_embedder(caption_emb)
        text_emb_for_nodes = projected_text_emb[batch_node]

        t = t.float()
        t = self.t_embedder(t)

        for i in range(self.depth):
            x = self.blocks[i](x, edge_index, t, text_emb_for_nodes)

            if controlnet_residuals is not None:
                if i < len(controlnet_residuals):
                    x = x + controlnet_residuals[i]

        x = self.final_layer(x, t)
        return x