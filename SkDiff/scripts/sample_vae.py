import torch
import sys
import os
import argparse
from pathlib import Path

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.misc import load_config, seed_all, edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d
from utils.transform import FeaturizeGraph
from models.vae import NodeCoordVAE
from utils.dataset import get_dataset


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
    # featurizer = FeaturizeGraph(use_rotate=True)
    transform = Compose([
        featurizer,  
    ])

    data_mode = cfg.val.data_mode
    val_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode=data_mode
    )
    print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
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
    vae_ckpt = torch.load(vae_ckpt_path, map_location=args.device, weights_only=False)
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


    with torch.no_grad():
        vae.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            batch = batch.to(args.device)
            edge_index = batch.bond_index
            node_pos = batch.node_pos
            batch_index = batch.batch  # 用于区分样本

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_recon, mu, logvar = vae(node_pos, edge_index, random=use_random)
                # print(torch.max(x_recon), torch.min(x_recon))

            # 这里只可视化第一个样本（用于拼图）
            mask = (batch_index == 0)
            node_pos_sample = node_pos[mask]
            x_recon_sample = x_recon[mask]
            edge_index_sample = edge_index[:, (batch_index[edge_index[0]] == 0) & (batch_index[edge_index[1]] == 0)]

            hierarchy = edge_index_to_hierarchy(edge_index_sample)
            joint_dict_gt = build_joint_dict(node_pos_sample[:, :3])       # {joint0: [x,y,z], ...}
            joint_dict_recon = build_joint_dict(x_recon_sample[:, :3])

            recon_loss = F.mse_loss(node_pos, x_recon)
            print('recon_loss: ', recon_loss)

            # 拼图：3视角 × 2列（GT / Recon）
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            proj_axes_list = [(0, 1), (1, 2), (0, 2)]
            proj_names = ['XY', 'YZ', 'XZ']

            for row_idx, (proj_axes, name) in enumerate(zip(proj_axes_list, proj_names)):
                visualize_skeleton_2d(joint_dict_gt, hierarchy, ax=axes[row_idx, 0],
                                    proj_axes=proj_axes, title=f"GT - {name}", color='blue')
                visualize_skeleton_2d(joint_dict_recon, hierarchy, ax=axes[row_idx, 1],
                                    proj_axes=proj_axes, title=f"Recon - {name}", color='green')

            
            if use_random:
                os.makedirs(f"vis/vae/{data_mode}/seed_{seed}/", exist_ok=True)
                plt.tight_layout()
                plt.savefig(f"vis/vae/{data_mode}/seed_{seed}/comparison_{i}.png", bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Saved vis/vae/{data_mode}/seed_{seed}/comparison_{i}.png")
            else:
                os.makedirs(f"vis/vae/{data_mode}/no_random/", exist_ok=True)
                plt.tight_layout()
                plt.savefig(f"vis/vae/{data_mode}/no_random/comparison_{i}.png", bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Saved vis/vae/{data_mode}/no_random/comparison_{i}.png")
            
            # if i == 3:
            #     break
