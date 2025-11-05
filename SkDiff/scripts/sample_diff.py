import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


from transformers import CLIPTokenizer, CLIPTextModel

from utils.data_loader import DataLoader as PyRenderDataLoader
from utils.pyrender_wrapper import PyRenderWrapper
from utils.misc import load_config, seed_all, edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d, visualize_skeleton_3d, save_skeleton_file, process_single_model_with_projections
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset
from models.latent_model import TextLatentModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sample/sample_diff.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="vis/diff", help="Directory for saving visualization images.")
    parser.add_argument("--skeleton_output_base_dir", type=str, default="sample_sk/sample_sk_diff", help="Base directory for saving generated skeleton files.")

    args = parser.parse_args()

    # === Load YAML ===
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem

    print(f"Loaded configuration from: {args.config}")
    print(f"Using device: {args.device}")
    
    # Set seet
    seed = int(args.seed) if args.seed else cfg.val.seed
    seed_all(seed)
    print(f"Using seed: {seed}")

    # Transforms
    featurizer = FeaturizeGraph(use_rotate=False, use_perturb=False)
    transform = Compose([
        featurizer,  
    ])

    # === Data ===
    data_mode = cfg.val.data_mode
    val_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode=data_mode
    )
    if hasattr(cfg.val, "default_view"):
        val_dataset.default_view = cfg.val.default_view
    if hasattr(cfg.val, "use_view_tag"):
        val_dataset.use_view_tag = cfg.val.use_view_tag
    print(f"Loaded {len(val_dataset)} samples from dataset with {val_dataset.default_view} view")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            follow_batch=featurizer.follow_batch)
    
    # === CLIP Loading (Freeze) ===
    print("Initializing CLIP model for captions...")
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    if not clip_ckpt_path or not os.path.exists(clip_ckpt_path):
        clip_ckpt_path = None
    clip_model_name = 'openai/clip-vit-large-patch14'
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name).to(args.device)

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print("CLIP loaded and frozen.")
    empty_ids = tokenizer("", return_tensors="pt", padding='max_length', truncation=True).input_ids.to(args.device)
    null_emb  = text_encoder(empty_ids).last_hidden_state

    # === VAE Loading (Freeze) ===
    print("Loading and freezing VAE...")
    vae = NodeCoordVAE(
            cfg.model.vae.coord_dim,
            cfg.model.vae.hidden_dim,
            cfg.model.vae.latent_dim,
            cfg.model.vae.norm_type
        ).to(args.device)
    if vae.latent_dim != cfg.model.diffusion.latent_dim:
        raise ValueError(f"VAE latent_dim ({vae.latent_dim}) must match diffusion latent_dim ({cfg.model.diffusion.latent_dim})")
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
        print("VAE weights loaded.")
    else:
        raise KeyError(f"Could not find VAE state dict in checkpoint: {vae_ckpt_path}")

    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    print("VAE loaded and frozen.")

    # === Model ===
    print("Building models...")
    model = TextLatentModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.heads,
    ).to(args.device)

    print("Initializing Gaussian Diffusion Wrapper...")
    diffusion = GaussianDiffusion(
        base_model=model,
        timesteps=cfg.model.diffusion.num_steps,
        sampling_timesteps=cfg.model.diffusion.num_steps,
        objective=cfg.model.diffusion.objective,
        beta_schedule="cosine",
        min_snr_loss_weight=cfg.model.diffusion.min_snr_loss_weight,
        min_snr_gamma=cfg.model.diffusion.min_snr_gamma,
        controlnet_model=None,
        device=args.device,
    )

    print(f"Loading diffusion model weights from: {cfg.model.diffusion.ckpt_path}")
    model.load_state_dict(torch.load(cfg.model.diffusion.ckpt_path, weights_only=False)['model'])

    use_text = cfg.val.use_text
    sub_folder_name = None
    if use_text:
        guidance_scale = cfg.model.diffusion.get('cfg_scale', 7.0)
        sub_folder_name = "use_text"
    else: 
        guidance_scale = 1.0
        sub_folder_name = "wo_text"

    vis_output_dir_arg = Path(cfg.get("output_dir", args.output_dir))
    skel_output_base_dir_arg = Path(cfg.get("skeleton_output_base_dir", args.skeleton_output_base_dir))
    vis_output_dir_arg.mkdir(parents=True, exist_ok=True)
    skel_output_base_dir_arg.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualization images to base: {vis_output_dir_arg}")
    print(f"Saving generated skeletons to base: {skel_output_base_dir_arg}")

    skeleton_output_dir = skel_output_base_dir_arg
    vis_output_dir = vis_output_dir_arg
    skeleton_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated skeletons to: {skeleton_output_dir}")
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {vis_output_dir}")

    loader = PyRenderDataLoader()
    renderer = PyRenderWrapper((512, 512))
    view_params = {
        "cam_pos_offset": np.array([0.7, 0.5, 0.7], dtype=np.float32) * 0.6, 
        "up_vector": [0, 1, 0]
    }

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            
            if not hasattr(batch, 'rig_path') or not batch.rig_path:
                print(f"WARNING: Batch {i} missing 'rig_path'. Using default filename.")
                original_filename_stem = f"unknown_sample_{i}" # Use stem for flexibility
            else:
                # Get only the filename without extension
                original_filename_stem = Path(batch.rig_path[0]).stem
            skeleton_save_path = skeleton_output_dir / f"{original_filename_stem}.txt"
            vis_save_path = vis_output_dir / f"{original_filename_stem}.png"

            if os.path.exists(vis_save_path):
                continue

            batch = batch.to(args.device)
            node_xy = batch.node_pos[:, :2]

            caption = batch.caption
            view = batch["view"][0]

            text_inputs = tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True)
            input_ids = text_inputs["input_ids"].to(args.device)
            attention_mask = text_inputs["attention_mask"].to(args.device) 
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            context = outputs.last_hidden_state

            null_embed_batch = null_emb.to(context.device)
            null_embed_batch = null_embed_batch.expand(context.size(0), -1, -1)

            print(f"\n--- Sampling Sample {i+1} (Base Name: {original_filename_stem}) ---")
            print(f"Graph nodes: {batch.num_nodes}, edges: {batch.bond_index.shape[1]}")
            
            node_emb = diffusion.sample(
                num_nodes=batch.num_nodes,
                edge_index=batch.bond_index,
                batch_node=batch.batch,
                num_graphs=batch.num_graphs,
                node_xy=node_xy,
                context=context,
                null_text_emb=null_embed_batch,
                cfg_scale_text=cfg.model.diffusion.cfg_scale,
                controlnet_cond=None,
                controlnet_conditioning_scale=None,
                cfg_scale_control=None,
                clip_denoised=False,
            )

            recon_pos = vae.decode(node_emb, batch.bond_index)
            recon_loss = F.mse_loss(recon_pos, batch.node_pos, reduction='mean')
            recon_loss_xy = F.mse_loss(recon_pos[:, :2], batch.node_pos[:, :2], reduction='mean')
            print(f"Sample {i+1} | Recon Loss (MSE): {recon_loss.item():.6f} | Recon Loss XY (MSE): {recon_loss_xy.item():.6f}")

            recon_pos = recon_pos * batch.scale + batch.translation
            if view == "yz":
                recon_pos = recon_pos[..., [2, 1, 0]]
            elif view == "xz":
                recon_pos = recon_pos[..., [0, 2, 1]]

            save_skeleton_file(skeleton_save_path, recon_pos, batch.bond_index)
            print(f"✅ Saved generated skeleton to {skeleton_save_path}")

            hierarchy = edge_index_to_hierarchy(batch.bond_index)
            joint_dict_gt = build_joint_dict(batch.node_pos[:, :3])       # {joint0: [x,y,z], ...}
            joint_dict_recon = build_joint_dict(recon_pos[:, :3])

            # (2D + 3D) x (GT + Recon)
            fig = plt.figure(figsize=(10, 10))
            proj_axes_map = {"xy": (0, 1), "yz": (2, 1), "xz": (0, 2)}

            visualize_skeleton_2d(joint_dict_gt, hierarchy, ax=fig.add_subplot(221), proj_axes=proj_axes_map["xy"], title=f"GT - {view}", color='blue')
            visualize_skeleton_2d(joint_dict_recon, hierarchy, ax=fig.add_subplot(222), proj_axes=proj_axes_map[view], title=f"Recon - {view}", color='green')

            if hasattr(batch, "mesh_path") and os.path.exists(batch.mesh_path[0]):
                process_single_model_with_projections(loader, renderer, batch.mesh_path[0], batch.rig_path[0], view_params, ax=fig.add_subplot(223), title="GT")
                process_single_model_with_projections(loader, renderer, batch.mesh_path[0], skeleton_save_path, view_params, ax=fig.add_subplot(224), title="Recon")
            else:
                visualize_skeleton_3d(joint_dict_gt, hierarchy, ax=fig.add_subplot(223, projection='3d'), title=f"GT - {view}", color='blue')
                visualize_skeleton_3d(joint_dict_recon, hierarchy, ax=fig.add_subplot(224, projection='3d'), title=f"Recon - {view}", color='green')

            plt.suptitle(f"Use text: {use_text}, Sample {i} (Guidance: {guidance_scale}) - Recon MSE: {recon_loss.item():.4f}, XY MSE: {recon_loss_xy.item():.4f}, \n{batch.caption[0]}", y=1.02)
            plt.tight_layout()
            plt.savefig(vis_save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved visualization image to {vis_save_path}")
           
            # break