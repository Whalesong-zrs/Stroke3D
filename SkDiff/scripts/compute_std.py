import torch
import random
import numpy as np
import sys
import yaml
import os
import argparse
import shutil
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.nn.functional as F

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from utils.misc import load_config, seed_all, edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset
from utils.train import get_optimizer, get_scheduler, inf_iterator
from models.graph_model import GraphLatentModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sample/sample_diff.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    # Set seet
    use_random = config.val.use_random
    seed = config.val.seed
    seed_all(seed=seed)

    # Transforms
    featurizer = FeaturizeGraph(use_rotate=False)
    transform = Compose([
        featurizer,  
    ])

    data_mode = config.val.data_mode
    val_dataset = get_dataset(
        config = config.dataset,
        transform = transform,
        mode=data_mode
    )

    val_loader = DataLoader(val_dataset, batch_size=config.val.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch)
    
    # vae freeze
    vae = NodeCoordVAE(
            config.model.vae.coord_dim, # N*3
            config.model.vae.hidden_dim, # 128
            config.model.vae.latent_dim, # 32
            config.model.vae.norm_type # layer
        ).to(args.device)
    vae.encoder.load_state_dict(torch.load(config.model.vae.ckpt_path)['encoder'])
    vae.decoder.load_state_dict(torch.load(config.model.vae.ckpt_path)['decoder'])
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    

    with torch.no_grad():

        mu_list = []
        z_list = []

        num_graphs = 0
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            batch = batch.to(args.device)
            num_nodes = batch.num_nodes
            edge_index = batch.bond_index 
            node_pos = batch.node_pos
            num_graphs += batch.num_graphs

            mu, logvar = vae.encode(node_pos, edge_index)
            z_start = vae.reparameterize(mu, logvar)

            mu_list.append(mu.cpu())
            z_list.append(z_start.cpu())
    print(len(z_list), num_graphs)
     # 收集完成后，拼接所有batch
    mu_all = torch.cat(mu_list, dim=0)  # (总节点数, latent_dim)
    z_all = torch.cat(z_list, dim=0)    # (总节点数, latent_dim)

    # 计算均值和标准差
    mu_mean = mu_all.mean()
    mu_std = mu_all.std()
    z_mean = z_all.mean()
    z_std = z_all.std()

    print(f"mu mean: {mu_mean:.6f}, mu std: {mu_std:.6f}")
    print(f"z mean: {z_mean:.6f}, z std: {z_std:.6f}")

    # 根据 reparameterize 后的 z 来算 scaling factor
    scaling_factor = 1.0 / z_std.item()
    print(f"Recommended scaling_factor: {scaling_factor:.6f}")

    # === 新增：每一维的std分析 ===
    std_per_dim = z_all.std(dim=0)  # shape: (latent_dim,)
    mean_std_per_dim = std_per_dim.mean().item()
    min_std_per_dim = std_per_dim.min().item()
    max_std_per_dim = std_per_dim.max().item()

    print(f"Per-dimension std - mean: {mean_std_per_dim:.6f}, min: {min_std_per_dim:.6f}, max: {max_std_per_dim:.6f}")

    # 画图
    plt.figure(figsize=(8,4))
    plt.plot(std_per_dim.numpy(), marker='o')
    plt.title('Per-dimension std of VAE Latent Space')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Std')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vae_latent_std_per_dim.png')
    
    threshold = 0.1
    num_low_std_dims = (std_per_dim < threshold).sum().item()
    print(f"Number of low-variance dimensions (std < {threshold}): {num_low_std_dims}/{std_per_dim.shape[0]}")

    from sklearn.decomposition import PCA
    import numpy as np

    # z_all shape: [num_nodes, latent_dim]
    z_np = z_all.numpy()

    # 做PCA
    pca = PCA(n_components=z_np.shape[1])  # 保留全部主成分
    z_pca = pca.fit_transform(z_np)

    explained_var_ratio = pca.explained_variance_ratio_

    # 画 explained variance ratio
    plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(explained_var_ratio), marker='o')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vae_pca_cumulative_variance.png')
  

    # 打印前几个主成分的贡献
    for i in range(32):
        print(f"PCA Component {i+1}: {explained_var_ratio[i]:.4f} (cumulative: {np.cumsum(explained_var_ratio)[i]:.4f})")

    # 可视化前2维投影
    plt.figure(figsize=(5,5))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.4, s=10)
    plt.title('VAE Latents - PCA Projection (First 2 Components)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    plt.savefig('vae_pca_scatter.png')
