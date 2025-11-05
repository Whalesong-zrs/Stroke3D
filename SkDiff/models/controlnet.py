from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.dense.linear import Linear
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

def zero_module(module):
    """Initialize parameters to zero."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
# ------------------------------------------------------

class ControlNetGraphModel(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 depth: int,
                 conditioning_channels: int,
                 clip_embed_dim: int,
                 dropout: float,
                 heads: int,
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        self.sinu_pos_emb = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.film_gamma = nn.ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.film_beta = nn.ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])

        self.input_proj = Linear(latent_dim, hidden_dim)
        self.clip_proj = Linear(clip_embed_dim, hidden_dim)

        self.controlnet_cond_embedding = zero_module(
            nn.Linear(conditioning_channels, hidden_dim)
        )

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.controlnet_blocks = nn.ModuleList([
            zero_module(Linear(hidden_dim, hidden_dim)) for _ in range(depth)
        ])

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                t: torch.Tensor,
                batch_node: torch.Tensor,
                num_graphs: int,
                caption_emb: Optional[torch.Tensor] = None,
                controlnet_cond: torch.Tensor = None,
                conditioning_scale: float = 1.0
               ) -> list[torch.Tensor]:
        
        # === 1. 时间编码 ===
        # t_emb 应该只针对原始节点 x 计算，因为caption节点的时间嵌入是特殊处理的（例如全零）
        # N_orig 是原始图中节点的数量
        N_orig = x.shape[0]
        if t.numel() == 1:
            if x.device != t.device: t = t.to(x.device)
            t_for_orig_nodes = t.repeat(N_orig)
        elif t.shape[0] != N_orig:
            raise ValueError(f"Shape mismatch: x has {N_orig} nodes, but t has {t.shape[0]} entries.")
        else:
            t_for_orig_nodes = t

        t_for_orig_nodes = t_for_orig_nodes.float()
        t_emb_orig = self.sinu_pos_emb(t_for_orig_nodes)
        t_emb_orig = self.time_mlp(t_emb_orig) # [N_orig, hidden_dim]

        # === 2. 初始特征和ControlNet条件投影 (仅针对原始节点) ===
        if controlnet_cond is not None and controlnet_cond.device != x.device:
            controlnet_cond = controlnet_cond.to(x.device)

        x_projected = self.input_proj(x) # [N_orig, hidden_dim]

        if controlnet_cond is not None:
            cond_emb = self.controlnet_cond_embedding(controlnet_cond) # [N_orig, hidden_dim]
            x_projected = x_projected + cond_emb
        
        # current_edge_index 用于 GNN 层，初始为原始的 edge_index
        current_edge_index = edge_index

        # === 3. Caption 处理 (如果提供) ===
        has_caption_nodes = False
        if caption_emb is not None and caption_emb.ndim > 0 and caption_emb.shape[0] > 0:
            has_caption_nodes = True
            if caption_emb.device != x_projected.device:
                caption_emb = caption_emb.to(x_projected.device)
            
            mapped_caption_emb = self.clip_proj(caption_emb) # [B, hidden_dim]

            # 合并原始节点特征和caption节点特征
            x_processed = torch.cat((x_projected, mapped_caption_emb), dim=0) # [N_orig + B, hidden_dim]

            # --- 构建新的边 (与 GraphLatentModel 镜像) ---
            new_edges_source = []
            new_edges_target = []
            caption_node_start_index = N_orig # 第一个caption节点的索引
            for i in range(num_graphs):
                nodes_in_graph_i_mask = (batch_node == i)
                original_nodes_indices_in_graph_i = torch.where(nodes_in_graph_i_mask)[0]
                if len(original_nodes_indices_in_graph_i) > 0:
                    caption_node_global_idx = caption_node_start_index + i
                    for node_idx in original_nodes_indices_in_graph_i:
                        new_edges_source.extend([caption_node_global_idx, node_idx.item()])
                        new_edges_target.extend([node_idx.item(), caption_node_global_idx])
            
            if new_edges_source:
                new_edges = torch.tensor([new_edges_source, new_edges_target], dtype=torch.long, device=x.device)
                current_edge_index = torch.cat([edge_index, new_edges], dim=1)
            # --- 结束构建新边 ---

            # --- 构建完整的时间嵌入 (与 GraphLatentModel 镜像) ---
            # caption 节点的时间嵌入设为零
            t_emb_for_captions = torch.zeros(caption_emb.shape[0], self.hidden_dim, device=x.device)
            final_t_emb = torch.cat([t_emb_orig, t_emb_for_captions], dim=0) # [N_orig + B, hidden_dim]
            # --- 结束构建时间嵌入 ---
        else: # 无caption或空caption
            x_processed = x_projected # 只有原始节点
            final_t_emb = t_emb_orig    # 只使用原始节点的时间嵌入
        
        # === 4. ControlNet 图 Transformer 块 ===
        h_intermediate = x_processed # 这是进入第一个GNN块的特征，会迭代更新
        collected_residuals_unscaled = [] 

        for i in range(self.depth):
            # block_input 是当前GNN块的输入，用于残差连接
            block_input = h_intermediate 
        
            # --- 图卷积/注意力 ---
            # self.convs[i] 内部也可能有 dropout
            conv_out = self.convs[i](block_input, current_edge_index) 
        
            # --- FiLM 调制 ---
            # final_t_emb 的维度应与 conv_out 的节点维度匹配
            gamma = self.film_gamma[i](final_t_emb) 
            beta = self.film_beta[i](final_t_emb)   
            modulated_out = gamma * conv_out + beta
        
            # --- 添加残差连接 (ResNet风格) ---
            h_after_residual_sum = modulated_out + block_input 
        
            # --- LayerNorm, GELU, Dropout ---
            h_after_norm = self.lns[i](h_after_residual_sum)
            h_after_activation = F.gelu(h_after_norm)
            # 注意：这里的dropout是块末尾的dropout，与TransformerConv内部的dropout是分开的
            h_intermediate = F.dropout(h_after_activation, p=self.dropout, training=self.training) 
        
            # --- 计算此块的控制残差 ---
            # control_res_full 的维度是 [N_orig (+ B if caption), hidden_dim]
            control_res_full = self.controlnet_blocks[i](h_intermediate) 
            
            # --- 关键：只收集原始节点的残差 ---
            if has_caption_nodes:
                # x_projected.shape[0] (即 N_orig) 是原始节点的数量
                control_res_original_nodes_only = control_res_full[:N_orig]
                collected_residuals_unscaled.append(control_res_original_nodes_only)
            else:
                collected_residuals_unscaled.append(control_res_full)
            
            # h_intermediate 已更新，将作为下一个GNN块的输入

        # === 5. 缩放最终收集的残差 ===
        # scaled_residuals 中的每个张量维度都是 [N_orig, hidden_dim]
        scaled_residuals = [res * conditioning_scale for res in collected_residuals_unscaled]

        return scaled_residuals