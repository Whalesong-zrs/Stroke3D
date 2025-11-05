import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

torch.multiprocessing.set_sharing_strategy('file_system')

# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from transformers import CLIPTokenizer, CLIPTextModel

from utils.data_loader import DataLoader as PyRenderDataLoader
from utils.pyrender_wrapper import PyRenderWrapper
from utils.misc import load_config, seed_all, edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d, save_skeleton_file, process_single_model_with_projections
from utils.transform import FeaturizeGraph
from utils.dataset import get_dataset
from models.latent_model import TextLatentModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE

methods = [
    {
        "caption": "DDPM, 500 steps",
        "name": "ddpm",
        "args": {"steps": 500}
    },
    {
        "caption": "DDPM, 1000 steps",
        "name": "ddpm",
        "args": {"steps": 1000}
    },
    {
        "caption": "DDIM, 500 steps",
        "name": "ddim",
        "args": {"steps": 500}
    },
    {
        "caption": "DDIM, 1000 steps",
        "name": "ddim",
        "args": {"steps": 1000}
    },
    {
        "caption": "DPM-Solver, 500 steps, order 2",
        "name": "dpm_solver",
        "args": {"steps": 500, "order": 2}
    },
    {
        "caption": "DPM-Solver, 1000 steps, order 2",
        "name": "dpm_solver",
        "args": {"steps": 1000, "order": 2}
    },
    {
        "caption": "DPM-Solver, 500 steps, order 3",
        "name": "dpm_solver",
        "args": {"steps": 500, "order": 3}
    },
    {
        "caption": "DPM-Solver, 1000 steps, order 3",
        "name": "dpm_solver",
        "args": {"steps": 1000, "order": 3}
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sample/sample_compare.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="vis/diff", help="Directory for saving visualization images.")
    parser.add_argument("--skeleton_output_base_dir", type=str, default="sample_sk/sample_sk_diff", help="Base directory for saving generated skeleton files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.val.seed)

    featurizer = FeaturizeGraph(use_rotate=False, use_perturb=False)
    transform = Compose([featurizer])
    data_mode = cfg.val.data_mode
    val_dataset = get_dataset(config=cfg.dataset, transform=transform, mode=data_mode)
    val_dataset.random_view = False
    val_dataset.random_tag = False
    val_dataset.random_flip = False
    if hasattr(cfg.val, "default_view"):
        val_dataset.default_view = cfg.val.default_view

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, follow_batch=featurizer.follow_batch)

    # CLIP
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    clip_model_name = 'openai/clip-vit-large-patch14'
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name).to(args.device)
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    empty_ids = tokenizer("", return_tensors="pt", padding='max_length', truncation=True).input_ids.to(args.device)
    null_emb = text_encoder(empty_ids).last_hidden_state

    # VAE
    vae = NodeCoordVAE(
        cfg.model.vae.coord_dim,
        cfg.model.vae.hidden_dim,
        cfg.model.vae.latent_dim,
        cfg.model.vae.norm_type
    ).to(args.device)
    vae_ckpt_path = cfg.model.vae.get('ckpt_path', None)
    vae_ckpt = torch.load(vae_ckpt_path, map_location=args.device, weights_only=False)
    vae_state_dict = vae_ckpt.get('model_state_dict', vae_ckpt.get('model', vae_ckpt))
    if vae_state_dict:
        if 'encoder' in vae_state_dict and 'decoder' in vae_state_dict:
            vae.encoder.load_state_dict(vae_state_dict['encoder'])
            vae.decoder.load_state_dict(vae_state_dict['decoder'])
        else:
            vae.load_state_dict(vae_state_dict)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    # Model
    model = TextLatentModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.heads,
    ).to(args.device)
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
    model.load_state_dict(torch.load(cfg.model.diffusion.ckpt_path, weights_only=False)['model'])

    vis_output_dir = Path(cfg.get("output_dir", args.output_dir))
    skel_output_base_dir = Path(cfg.get("skeleton_output_base_dir", args.skeleton_output_base_dir))
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    skel_output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualization images to base: {vis_output_dir}")
    print(f"Saving generated skeletons to base: {skel_output_base_dir}")

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
                original_filename_stem = f"unknown_sample_{i}"
            else:
                original_filename_stem = Path(batch.rig_path[0]).stem

            batch = batch.to(args.device)
            node_xy = batch.node_pos[:, :2]
            caption = batch.caption
            view = batch["view"][0]
            uuid = batch["uuid"][0]

            text_inputs = tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True)
            input_ids = text_inputs["input_ids"].to(args.device)
            attention_mask = text_inputs["attention_mask"].to(args.device)
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            context = outputs.last_hidden_state
            null_embed_batch = null_emb.to(context.device).expand(context.size(0), -1, -1)

            # 多种采样方法对比
            for method in methods:
                input_dict = {
                    "num_nodes": batch.num_nodes,
                    "edge_index": batch.bond_index,
                    "batch_node": batch.batch,
                    "num_graphs": batch.num_graphs,
                    "node_xy": node_xy,
                    "context": context,
                    "null_text_emb": null_embed_batch,
                    "cfg_scale_text": cfg.model.diffusion.cfg_scale,
                    "controlnet_cond": None,
                    "controlnet_conditioning_scale": None,
                    "cfg_scale_control": None,
                    "clip_denoised": False,
                    **method["args"]
                }

                if method["name"] == "ddpm":
                    node_emb = diffusion.sample(**input_dict)
                elif method["name"] == "ddim":
                    node_emb = diffusion.sample_ddim(**input_dict)
                elif method["name"] == "dpm_solver":
                    node_emb = diffusion.sample_dpm_solver(**input_dict)
                else:
                    raise ValueError(f"Unknown sampling method: {method["name"]}")

                recon_pos = vae.decode(node_emb, batch.bond_index)
                recon_pos = recon_pos * batch.scale + batch.translation
                method["recon_pos"] = recon_pos

            cols = 2
            rows = (len(methods) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            axes = axes.flatten()

            for i, method in enumerate(methods):
                joint_dict_pred = build_joint_dict(method["recon_pos"])
                skeleton_save_path = skel_output_base_dir / f"{original_filename_stem}_{method['caption'].replace(' ', '_').replace(',', '_')}.txt"
                save_skeleton_file(skeleton_save_path, method["recon_pos"], batch.bond_index)
                process_single_model_with_projections(loader, renderer, batch.mesh_path[0], skeleton_save_path, view_params, ax=axes[i], title=method["caption"])

            plt.suptitle(f"Sample {uuid}\n{caption[0]}", y=1.02)
            plt.tight_layout()
            save_path = vis_output_dir / f"{original_filename_stem}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"\n✅ Saved comparison image to {save_path}")