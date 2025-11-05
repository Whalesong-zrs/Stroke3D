import torch
import numpy as np
import sys
import os
import argparse
from pathlib import Path

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
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np

from utils.misc import load_config, seed_all, compute_frechet_distance, kl_divergence_gaussians, collect_vae_embeddings
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset
from models.vae import NodeCoordVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sample/sample_vae.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()

    # === Load YAML ===
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem

    use_random = cfg.val.use_random
    seed = cfg.val.seed
    seed_all(seed=seed)

    # Transforms
    featurizer = FeaturizeGraph(use_rotate=False)
    transform = Compose([
        featurizer,  
    ])

    train_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode='train'
    )
    val_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode='val'
    )
    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=False,
                            follow_batch=featurizer.follow_batch)
    
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False,
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
    
    vae_train_emb = collect_vae_embeddings(vae, train_loader, args.device)
    vae_val_emb = collect_vae_embeddings(vae, val_loader, args.device)

    z1 = vae_train_emb.numpy()
    z2 = vae_val_emb.numpy()

    mu1, cov1 = np.mean(z1, axis=0), np.cov(z1, rowvar=False)
    mu2, cov2 = np.mean(z2, axis=0), np.cov(z2, rowvar=False)

    fid = compute_frechet_distance(mu1, cov1, mu2, cov2)
    kl = kl_divergence_gaussians(mu1, cov1, mu2, cov2)

    print(f'fid: {fid}, kl: {kl}')
