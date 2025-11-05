# --- START OF FILE sample_controlnet2.py (MODIFIED FOR V2, NO LOGGER, SEPARATE SAVE PATHS) ---

import torch
import random
import numpy as np
import sys
import yaml
import os
import argparse
import shutil
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import torch.nn.functional as F
# import logging # REMOVED

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
from transformers import CLIPTokenizer, CLIPTextModel
# Import utilities
from utils.misc import load_config, seed_all, edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d, save_skeleton_file
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset

from models.graph_model import GraphLatentModel
from models.controlnet import ControlNetGraphModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sample/sample_controlnet.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='vis/control', help="Directory for saving visualization images.")
    parser.add_argument('--skeleton_output_base_dir', type=str, default='sample_sk/sample_sk_control', help="Base directory for saving generated skeleton files.")
    args = parser.parse_args()

    # Load configs
    cfg = load_config(args.config)
    config_name = Path(args.config).stem

    print(f"Loaded configuration from: {args.config}")
    print(f"Using device: {args.device}")

    # Set seed
    seed = cfg.val.seed
    seed_all(seed)

    # Transforms
    featurizer = FeaturizeGraph(use_rotate=False)
    transform = Compose([featurizer])

    # Dataset and DataLoader
    data_mode = cfg.val.data_mode
    print(f"Loading dataset mode: {data_mode}")
    vis_dataset = get_dataset(
        config = cfg.dataset,
        transform = transform,
        mode=data_mode
    )

    follow_batch_keys = getattr(featurizer, 'follow_batch', [])
    vis_loader = DataLoader(
        vis_dataset,
        batch_size=1,
        shuffle=False,
        follow_batch=follow_batch_keys
    )
    print(f"Dataset size: {len(vis_dataset)}")

    # === CLIP Loading (Freeze) ===
    print("Initializing CLIP model for captions...")
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    clip_model_name = 'openai/clip-vit-large-patch14'
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name).to(args.device)

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print("CLIP loaded and frozen.")
    
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

    # === V2 Model Instantiation ===
    print("Building models...")
    # (Model instantiation code remains the same)
    # --- Base Model (Frozen) ---
    base_model = GraphLatentModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.get('heads', 4)
    ).to(args.device)
    # --- ControlNet Model (Will use EMA weights) ---
    if not hasattr(cfg.model.controlnet, 'conditioning_channels'):
         raise ValueError("Config missing 'model.controlnet.conditioning_channels'")
    controlnet_model = ControlNetGraphModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        conditioning_channels=cfg.model.controlnet.conditioning_channels,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=0,
        heads=cfg.model.diffusion.get('heads', 4)
    ).to(args.device)

    # === Weight Loading for Base and ControlNet ===
    # (Weight loading code for both Base and ControlNet remains the same)
    # --- Load Base Model Weights (Prioritize EMA) ---
    base_ckpt_path = cfg.model.diffusion.get('ckpt_path', None)
    if not base_ckpt_path or not os.path.isfile(base_ckpt_path):
        print(f"ERROR: Base model checkpoint path is required but not found or specified: {base_ckpt_path}")
        exit(1)
    print(f"Loading pre-trained base model weights from: {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location=args.device, weights_only=False)
    if 'ema' in base_ckpt:
        print("Found EMA state in base checkpoint. Applying EMA weights...")
        try:
            ema_decay = base_ckpt['ema'].get('decay', 0.9999)
            ema_base = ExponentialMovingAverage(base_model.parameters(), decay=ema_decay)
            ema_base.load_state_dict(base_ckpt['ema'])
            ema_base.copy_to(base_model.parameters())
            print("Successfully applied EMA weights to base model.")
        except Exception as e:
            print(f"ERROR: Failed to load or apply EMA state: {e}. Check EMA compatibility or checkpoint integrity.")
            print("Attempting to fall back to loading raw 'model' weights...")
            if 'model' in base_ckpt:
                base_model_state_dict = base_ckpt['model']
                load_result_base = base_model.load_state_dict(base_model_state_dict, strict=False)
                print(f"Base model raw loading result (fallback): {load_result_base}")
            else:
                raise KeyError("Neither 'ema' nor 'model' state found or loaded correctly in base checkpoint after EMA failure.")
    elif 'model' in base_ckpt:
        print("WARNING: EMA state not found in base checkpoint. Loading raw 'model' weights.")
        base_model_state_dict = base_ckpt['model']
        load_result_base = base_model.load_state_dict(base_model_state_dict, strict=False)
        print(f"Base model raw loading result: {load_result_base}")
    else:
        raise KeyError(f"Neither 'ema' nor 'model' state found in base checkpoint: {base_ckpt_path}")
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    print("Base model weights loaded, set to eval, and frozen.")

    # --- Load ControlNet Weights (Prioritize EMA) ---
    controlnet_ckpt_path = cfg.model.controlnet.get('ckpt_path', None)
    if not controlnet_ckpt_path or not os.path.isfile(controlnet_ckpt_path):
        print(f"ERROR: Trained ControlNet checkpoint path is required but not found or specified: {controlnet_ckpt_path}")
        exit(1)
    print(f"Loading trained ControlNet weights from: {controlnet_ckpt_path}")
    ctrl_ckpt = torch.load(controlnet_ckpt_path, map_location=args.device, weights_only=False)
    if 'ema' in ctrl_ckpt:
        print("Found EMA state in ControlNet checkpoint. Applying EMA weights...")
        try:
            ema_decay_ctrl = ctrl_ckpt['ema'].get('decay', 0.9999)
            ema_ctrl = ExponentialMovingAverage(controlnet_model.parameters(), decay=ema_decay_ctrl)
            ema_ctrl.load_state_dict(ctrl_ckpt['ema'])
            ema_ctrl.copy_to(controlnet_model.parameters())
            print("Successfully applied EMA weights to ControlNet model.")
        except Exception as e:
            print(f"ERROR: Failed to load or apply ControlNet EMA state: {e}.")
            print("Attempting to fall back to loading raw 'controlnet_model' or 'model' weights...")
            ctrl_model_state_dict = ctrl_ckpt.get('controlnet_model', ctrl_ckpt.get('model'))
            if ctrl_model_state_dict is not None:
                load_result_ctrl = controlnet_model.load_state_dict(ctrl_model_state_dict, strict=True)
                print(f"ControlNet raw loading result (fallback): {load_result_ctrl}")
            else:
                raise KeyError("Neither 'ema' nor 'controlnet_model'/'model' state found or loaded correctly in ControlNet checkpoint after EMA failure.")
    elif 'controlnet_model' in ctrl_ckpt or 'model' in ctrl_ckpt:
        print("WARNING: EMA state not found in ControlNet checkpoint. Loading raw model weights.")
        ctrl_model_state_dict = ctrl_ckpt.get('controlnet_model', ctrl_ckpt.get('model'))
        try:
            load_result_ctrl = controlnet_model.load_state_dict(ctrl_model_state_dict, strict=True)
            print(f"ControlNet raw loading result: {load_result_ctrl}")
        except RuntimeError as e:
             print(f"ERROR: Failed to load ControlNet raw weights with strict=True: {e}.")
             print("Attempting to load with strict=False...")
             load_result_ctrl = controlnet_model.load_state_dict(ctrl_model_state_dict, strict=False)
             print(f"ControlNet raw loading result (strict=False): {load_result_ctrl}")
    else:
        raise KeyError(f"Neither 'ema' nor 'controlnet_model'/'model' state found in ControlNet checkpoint: {controlnet_ckpt_path}")
    controlnet_model.eval()
    print("ControlNet model weights loaded and set to eval.")

    # === V2 Diffusion Wrapper Instantiation ===
    print("Initializing V2 Gaussian Diffusion Wrapper...")
    # (Diffusion wrapper instantiation remains the same)
    diffusion = GaussianDiffusion(
        base_model=base_model,
        timesteps=cfg.model.diffusion.num_steps,
        sampling_timesteps=cfg.model.diffusion.get('sampling_timesteps', cfg.model.diffusion.num_steps), # Allow separate sampling steps
        objective = cfg.model.diffusion.objective,
        beta_schedule = 'cosine',
        ddim_sampling_eta = cfg.model.diffusion.get('ddim_sampling_eta', 1.0),
        offset_noise_strength = cfg.model.diffusion.get('offset_noise_strength', 0.0),
        min_snr_loss_weight=cfg.model.diffusion.get('min_snr_loss_weight', True),
        min_snr_gamma=cfg.model.diffusion.get('min_snr_gamma', 5),
        controlnet_model=controlnet_model,
        device=args.device
    )
    print("Diffusion wrapper initialized.")

    controlnet_conditioning_scale = cfg.model.controlnet.get('controlnet_conditioning_scale', 1.0)
    use_text = cfg.val.use_text
    sub_folder_name = None
    if use_text:
        guidance_scale = cfg.model.diffusion.get('cfg_scale', 7.0) # Get guidance scale once
        sub_folder_name = "use_text"
    else: 
        guidance_scale = 1.0
        sub_folder_name = "wo_text"
     # Setup Output Directories
    vis_output_dir_arg = Path(args.output_dir) # Visualization dir from args
    skel_output_base_dir_arg = Path(args.skeleton_output_base_dir) # Skeleton base dir from args
    vis_output_dir_arg.mkdir(parents=True, exist_ok=True)
    skel_output_base_dir_arg.mkdir(parents=True, exist_ok=True) # Ensure skeleton base exists
    print(f"Saving visualization images to base: {vis_output_dir_arg}")
    print(f"Saving generated skeletons to base: {skel_output_base_dir_arg}")

    skeleton_output_dir = skel_output_base_dir_arg / data_mode / sub_folder_name / f"gs_{guidance_scale:.1f}"
    vis_output_dir = vis_output_dir_arg / data_mode / sub_folder_name / f"gs_{guidance_scale:.1f}"
    skeleton_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated skeletons to: {skeleton_output_dir}")
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {vis_output_dir}")

    # === Sampling Loop ===
    total_recon_loss = 0.0
    total_recon_xy_loss = 0.0
    samples_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(vis_loader, desc='Sampling')):

            if not hasattr(batch, 'file_path') or not batch.file_path:
                print(f"WARNING: Batch {i} missing 'file_path'. Using default filename.")
                original_filename_stem = f"unknown_sample_{i}" # Use stem for flexibility
            else:
                # Get only the filename without extension
                original_filename_stem = Path(batch.file_path[0]).stem

            batch = batch.to(args.device)
            x_gt_pos = batch.node_pos # Ground truth positions
            control_conditioning_channels = cfg.model.controlnet.conditioning_channels
            control_signal = x_gt_pos[:, :control_conditioning_channels].clone().detach()

            text_inputs = tokenizer(batch.captions, return_tensors="pt", padding='max_length', truncation=True)
            input_ids = text_inputs["input_ids"].to(args.device)
            attention_mask = text_inputs["attention_mask"].to(args.device) 
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            if use_text:
                use_caption_emb = outputs.pooler_output
            else:
                use_caption_emb = torch.zeros_like(outputs.pooler_output)

            print(f"\n--- Sampling Sample {i+1} (Base Name: {original_filename_stem}) ---")
            print(f"Graph nodes: {batch.num_nodes}, edges: {batch.bond_index.shape[1]}")

            try:
                 node_emb_gen = diffusion.sample(
                    num_nodes=batch.num_nodes,
                    edge_index=batch.bond_index,
                    batch_node=batch.batch,
                    num_graphs=batch.num_graphs,
                    caption_emb=use_caption_emb,
                    cfg_scale=guidance_scale,
                    controlnet_cond=control_signal,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                 )

                 scaling_factor = cfg.model.diffusion.get('scaling_factor', 1.0)
                 node_emb_scaled = node_emb_gen / scaling_factor if scaling_factor != 1.0 else node_emb_gen

                 recon_pos = vae.decode(node_emb_scaled, batch.bond_index)

                 recon_loss = F.mse_loss(recon_pos, x_gt_pos, reduction='mean')
                 recon_loss_xy = F.mse_loss(recon_pos[:, :2], x_gt_pos[:, :2], reduction='mean')
                 print(f"Sample {i+1} | Recon Loss (MSE): {recon_loss.item():.6f} | Recon Loss XY (MSE): {recon_loss_xy.item():.6f}")

                 total_recon_loss += recon_loss.item()
                 total_recon_xy_loss += recon_loss_xy.item()
                 samples_processed += 1

                 # +++ SAVE THE GENERATED SKELETON +++
                 # Construct the output path using the original filename stem + .txt
                 skeleton_save_path = skeleton_output_dir / f"{original_filename_stem}.txt"
                 save_skeleton_file(skeleton_save_path, recon_pos, batch.bond_index)
                 print(f"✅ Saved generated skeleton to {skeleton_save_path}")
                 # ++++++++++++++++++++++++++++++++++++

                 # Visualization
                 hierarchy = edge_index_to_hierarchy(batch.bond_index)
                 joint_dict_gt = build_joint_dict(x_gt_pos[:, :3])
                 joint_dict_recon = build_joint_dict(recon_pos[:, :3])

                 fig, axes = plt.subplots(3, 2, figsize=(10, 12))
                 proj_axes_list = [(0, 1), (1, 2), (0, 2)]
                 proj_names = ['XY', 'YZ', 'XZ']

                 for row_idx, (proj_axes, name) in enumerate(zip(proj_axes_list, proj_names)):
                     visualize_skeleton_2d(joint_dict_gt, hierarchy, ax=axes[row_idx, 0],
                                         proj_axes=proj_axes, title=f"GT - {name}", color='blue')
                     visualize_skeleton_2d(joint_dict_recon, hierarchy, ax=axes[row_idx, 1],
                                         proj_axes=proj_axes, title=f"Recon (ControlNet) - {name}", color='green')
                     if row_idx == 0:
                          axes[row_idx, 1].scatter(control_signal[:, 0].cpu().numpy(), control_signal[:, 1].cpu().numpy(),
                                                  color='red', s=15, label='Control XY', alpha=0.6, zorder=5)
                          axes[row_idx, 1].legend()

                 plt.suptitle(f"Use text: {use_text},Sample {i} (Guidance: {guidance_scale}) - Recon MSE: {recon_loss.item():.4f}, XY MSE: {recon_loss_xy.item():.4f}, \n {batch.captions}", y=1.02)
                 plt.tight_layout()
                 plt.savefig(f'{vis_output_dir}/comparison_{i}.png', bbox_inches='tight')
                 plt.close(fig)
                 print(f"✅ Saved {vis_output_dir}/comparison_{i}.png")

            except Exception as e:
                print(f"ERROR: Error processing sample {i} (Base Name: {original_filename_stem}): {e}")
                import traceback
                traceback.print_exc()
                continue

    # Calculate average losses
    if samples_processed > 0:
        avg_recon_loss = total_recon_loss / samples_processed
        avg_recon_xy_loss = total_recon_xy_loss / samples_processed
        print(f"\n--- Sampling Summary ---")
        print(f"Processed {samples_processed} requested samples.")
        print(f"Average Reconstruction Loss (MSE): {avg_recon_loss:.6f}")
        print(f"Average XY Reconstruction Loss (MSE): {avg_recon_xy_loss:.6f}")
    else:
        print("WARNING: No samples were processed successfully.")

    print("Sampling script finished.")