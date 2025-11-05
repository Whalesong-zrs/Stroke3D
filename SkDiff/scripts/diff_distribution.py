import torch
import numpy as np
import sys
import os
import argparse
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.misc import load_config, seed_all, compute_frechet_distance, kl_divergence_gaussians, collect_vae_embeddings
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset
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
    cfg = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    # Set seet
    use_random = cfg.val.use_random
    seed = cfg.val.seed
    seed_all(seed=seed)

    # Transforms
    featurizer = FeaturizeGraph(use_rotate=False)
    transform = Compose([
        featurizer,  
    ])

    data_mode = cfg.val.data_mode
    val_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode=data_mode
    )

    val_loader = DataLoader(val_dataset, batch_size=cfg.val.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch)
    
     # === VAE Loading (Freeze) ===
    print("Loading and freezing VAE...")
    # (VAE Loading code remains the same)
    vae = NodeCoordVAE(
            cfg.model.vae.coord_dim,
            cfg.model.vae.hidden_dim,
            cfg.model.vae.latent_dim,
            cfg.model.vae.norm_type
        ).to(args.device)
    vae_ckpt_path = cfg.model.vae.get('ckpt_path', None)
    if not vae_ckpt_path or not os.path.isfile(vae_ckpt_path):
         raise FileNotFoundError(f"VAE checkpoint path not found or specified: {vae_ckpt_path}")
    print(f"Loading VAE weights from: {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=args.device)
    vae_state_dict = vae_ckpt.get('model_state_dict', vae_ckpt.get('model', vae_ckpt))
    if vae_state_dict:
        if 'encoder' in vae_state_dict and 'decoder' in vae_state_dict:
             vae.encoder.load_state_dict(vae_state_dict['encoder'])
             vae.decoder.load_state_dict(vae_state_dict['decoder'])
        else:
             vae.load_state_dict(vae_state_dict)
        print("VAE weights loaded successfully.")
    else:
        raise KeyError(f"Could not find VAE state dict in checkpoint: {vae_ckpt_path}")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE loaded and frozen.")
    vae_embed = collect_vae_embeddings(vae, val_loader, args.device)

    # === Model ===
    print("Building Diffusion model…")
    model = GraphLatentModel( 
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.get('heads', 4)
    ).to(args.device)
    
    print("Initializing Gaussian Diffusion Wrapper...")
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=cfg.model.diffusion.num_steps,
        sampling_timesteps=cfg.model.diffusion.num_steps,
        objective=cfg.model.diffusion.objective,
        beta_schedule="cosine",
        ddim_sampling_eta=1.0,
        offset_noise_strength=0.0,
        device=args.device,
    )

    # model.load_state_dict(torch.load(config.model.diffusion.ckpt_path)['model'])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ckpt = torch.load(cfg.model.diffusion.ckpt_path, map_location=args.device)
    ema.load_state_dict(ckpt['ema'])                 # ✅ 只加载 EMA
    ema.copy_to(model.parameters()) 

    with torch.no_grad():
        model.eval()

        z_diff_list = []

        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            batch = batch.to(args.device)
            z_sample = diffusion.sample(
                num_nodes=batch.num_nodes,
                edge_index=batch.bond_index,
                num_graphs=batch.num_graphs,
                batch_node=batch.batch
            )
            z_diff_list.append(z_sample.cpu())
            z_diff_all = torch.cat(z_diff_list, dim=0)  # [N_val_nodes, latent_dim]

    z_diff_all = torch.cat(z_diff_list, dim=0)
    
    print("=== VAE Encoder embedding Distribution ===")
    print("Mean:", vae_embed.mean())
    print("Std :", vae_embed.std())

    print("=== Diffusion z₀ Distribution ===")
    print("Mean:", z_diff_all.mean())
    print("Std :", z_diff_all.std())   
    
    z1 = vae_embed.numpy()
    z2 = z_diff_all.numpy()

    mu1, cov1 = np.mean(z1, axis=0), np.cov(z1, rowvar=False)
    mu2, cov2 = np.mean(z2, axis=0), np.cov(z2, rowvar=False)

    fid = compute_frechet_distance(mu1, cov1, mu2, cov2)
    kl = kl_divergence_gaussians(mu1, cov1, mu2, cov2)

    print("val/fid", fid)
    print("val/kl_div", kl)

    do_visualize = True
    if do_visualize:
        # 拼接数据
        z_all = torch.cat([vae_embed, z_diff_all], dim=0).cpu().numpy()
        labels = ['VAE'] * len(vae_embed) + ['Diffusion'] * len(z_diff_all)

        # 做PCA
        pca = PCA(n_components=2)
        z_vis = pca.fit_transform(z_all)

        # 划分索引
        vae_idx = np.array(labels) == 'VAE'
        diff_idx = np.array(labels) == 'Diffusion'

        # 创建子图：一行三列
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # 左图：只画VAE latent
        axs[0].scatter(z_vis[vae_idx, 0], z_vis[vae_idx, 1], alpha=0.5, label='VAE')
        axs[0].set_title('PCA: VAE Latents')
        axs[0].legend()
        axs[0].grid(True)

        # 中图：只画Diffusion latent
        axs[1].scatter(z_vis[diff_idx, 0], z_vis[diff_idx, 1], alpha=0.5, label='Diffusion', color='orange')
        axs[1].set_title('PCA: Diffusion Latents')
        axs[1].legend()
        axs[1].grid(True)

        # 右图：VAE和Diffusion一起画
        axs[2].scatter(z_vis[vae_idx, 0], z_vis[vae_idx, 1], alpha=0.3, label='VAE')
        axs[2].scatter(z_vis[diff_idx, 0], z_vis[diff_idx, 1], alpha=0.3, label='Diffusion', color='orange')
        axs[2].set_title('PCA: VAE vs Diffusion')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.title('PCA of Latent Distributions')
        plt.legend()
        plt.tight_layout()
        plt.savefig('vis/pca_vis.png')

    num_sample = 4000  # or 2000
    idx_vae  = torch.randperm(vae_embed.shape[0])[:num_sample]
    idx_diff = torch.randperm(z_diff_all.shape[0])[:num_sample]

    z_vae_sample  = vae_embed[idx_vae]
    z_diff_sample = z_diff_all[idx_diff]

    # ------------------------------------------------------------------
    # 2. t-SNE 降维
    # ------------------------------------------------------------------
    z_all  = torch.cat([z_vae_sample, z_diff_sample], dim=0).numpy()
    labels = ['VAE'] * len(z_vae_sample) + ['Diffusion'] * len(z_diff_sample)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        init='pca',
        random_state=42,
        verbose=1
    )
    z_tsne = tsne.fit_transform(z_all)

    # 放进 DataFrame 便于 seaborn 绘图
    df = pd.DataFrame(z_tsne, columns=['x', 'y'])
    df['label'] = labels

    # ------------------------------------------------------------------
    # 3. 画图：横向 1×3 子图
    # ------------------------------------------------------------------
    sns.set(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(24, 8),
        sharex=False, sharey=False
    )

    # --- (0) Combined ---
    sns.scatterplot(
        ax=axes[0],
        data=df, x='x', y='y',
        hue='label', style='label',
        s=60, alpha=0.6, linewidth=0.2
    )
    axes[0].set_title('t-SNE: VAE vs Diffusion', fontsize=14)
    axes[0].legend(title='Latent Source')

    # --- (1) VAE Only ---
    df_vae = df[df['label'] == 'VAE']
    sns.scatterplot(
        ax=axes[1],
        data=df_vae, x='x', y='y',
        color='tab:blue', s=60, alpha=0.6, linewidth=0.2
    )
    axes[1].set_title('t-SNE: VAE Latents', fontsize=14)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    # --- (2) Diffusion Only ---
    df_diff = df[df['label'] == 'Diffusion']
    sns.scatterplot(
        ax=axes[2],
        data=df_diff, x='x', y='y',
        color='tab:orange', s=60, alpha=0.6, linewidth=0.2
    )
    axes[2].set_title('t-SNE: Diffusion Latents', fontsize=14)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    fig.tight_layout()

    # ------------------------------------------------------------------
    # 4. 保存为 PNG
    # ------------------------------------------------------------------
    fig.savefig('vis/tsne_latents_3panel.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Saved figure -> tsne_latents_3panel.png')