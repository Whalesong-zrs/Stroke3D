import sys
import os
import shutil
import argparse
from pathlib import Path

from collections import defaultdict# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from tqdm import tqdm
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.data_loader import DataLoader as PyRenderDataLoader
from utils.pyrender_wrapper import PyRenderWrapper
from utils.misc import load_config, seed_all, get_new_log_dir, get_logger, \
    edge_index_to_hierarchy, build_joint_dict, visualize_skeleton_2d, visualize_skeleton_3d, save_skeleton_file, process_single_model_with_projections
from utils.transform import FeaturizeGraph
from utils.dataset import build_dataloader
from utils.train import get_optimizer, get_scheduler, inf_iterator, should_step_each_iter

from models.latent_model import TextLatentModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/train_parallel.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint (.pt)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # === Load YAML ===
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem

    # === DDP Initialize ===
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # === Seed & log dir ===
    seed_all(cfg.train.seed)
    if rank == 0:
        log_dir = get_new_log_dir(cfg.log_dir, prefix=cfg_name)
        ckpt_dir = Path(log_dir, "checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger("rank0", log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        shutil.copyfile(args.config, Path(log_dir, Path(args.config).name))
        logger.info("Using config %s", args.config)
        logger.info(f"Logging to: {log_dir}")
        logger.info(f"Using device: {device}")
    else:
        logger = get_logger(f"rank{rank}", None)
        logger.disabled = True
        writer = None

    # === Data ===
    featurizer = FeaturizeGraph(use_rotate=cfg.dataset.get('use_rotate', True))

    train_loader = build_dataloader(cfg, "train", featurizer, distributed=True, rank=rank, world_size=world_size)
    val_loader = build_dataloader(cfg, "val", featurizer, distributed=False, rank=rank, world_size=world_size)
    train_loader.dataset.use_tag = cfg.train.use_tag
    val_loader.dataset.use_tag = cfg.train.use_tag
    train_loader.sampler.set_epoch(0)
    train_iter = inf_iterator(train_loader, distributed=True)
    val_iter = inf_iterator(val_loader, distributed=False)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # === CLIP Loading (Freeze) ===
    logger.info("Initializing CLIP model for captions...")
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    if not clip_ckpt_path or not os.path.exists(clip_ckpt_path):
        clip_ckpt_path = None
    clip_model_name = 'openai/clip-vit-large-patch14'
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path if clip_ckpt_path else clip_model_name).to(device)

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    logger.info("CLIP loaded and frozen.")
    null_inputs = tokenizer("", return_tensors="pt", padding='max_length', truncation=True)
    null_ids = null_inputs["input_ids"].to(device)
    null_attention_mask = null_inputs["attention_mask"].to(device)
    null_outputs = text_encoder(input_ids=null_ids, attention_mask=null_attention_mask)
    null_emb = null_outputs.last_hidden_state

    # === VAE Loading (Freeze) ===
    logger.info("Loading and freezing VAE...")
    vae = NodeCoordVAE(
        cfg.model.vae.coord_dim,
        cfg.model.vae.hidden_dim,
        cfg.model.vae.latent_dim,
        cfg.model.vae.norm_type
    ).to(device)
    if vae.latent_dim != cfg.model.diffusion.latent_dim:
        raise ValueError(f"VAE latent_dim ({vae.latent_dim}) must match diffusion latent_dim ({cfg.model.diffusion.latent_dim})")
    vae_ckpt_path = cfg.model.vae.get('ckpt_path', None)
    if not vae_ckpt_path or not os.path.isfile(vae_ckpt_path):
         raise FileNotFoundError(f"VAE checkpoint path not found or specified: {vae_ckpt_path}")
    logger.info(f"Loading VAE weights from: {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae_state_dict = vae_ckpt.get('model_state_dict', vae_ckpt.get('model', vae_ckpt))
    if vae_state_dict:
        if 'encoder' in vae_state_dict and 'decoder' in vae_state_dict:
             vae.encoder.load_state_dict(vae_state_dict['encoder'])
             vae.decoder.load_state_dict(vae_state_dict['decoder'])
        else:
             vae.load_state_dict(vae_state_dict)
        logger.info("VAE weights loaded.")
    else:
        raise KeyError(f"Could not find VAE state dict in checkpoint: {vae_ckpt_path}")

    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    logger.info("VAE loaded and frozen.")

    # === Model ===
    logger.info("Building model...")
    model = TextLatentModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.heads,
    ).to(device)
    logger.info("Building distributed model...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    logger.info("Initializing Gaussian Diffusion Wrapper...")
    diffusion = GaussianDiffusion(
        base_model=model.module,
        timesteps=cfg.model.diffusion.num_steps,
        sampling_timesteps=cfg.model.diffusion.num_steps,
        objective=cfg.model.diffusion.objective,
        beta_schedule="cosine",
        min_snr_loss_weight=cfg.model.diffusion.min_snr_loss_weight,
        min_snr_gamma=cfg.model.diffusion.min_snr_gamma,
        controlnet_model=None,
        device=device,
    )

    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.module.parameters() if p.requires_grad))

    # === Optim & sched ===
    optimizer = get_optimizer(cfg.train.optimizer, model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    step_each_iter = should_step_each_iter(cfg.train.scheduler.type)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.train.use_amp)

    def train(it_idx: int):
        diffusion.train()
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iter).to(device)
        node_xy = batch.node_pos[:, :2]
        context = batch.get("clip_emb", None)

        with torch.no_grad():
            if context is None:
                text_inputs = tokenizer(batch.caption, return_tensors="pt", padding='max_length', truncation=True)
                input_ids = text_inputs["input_ids"].to(device)
                attention_mask = text_inputs["attention_mask"].to(device) 
                outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                context = outputs.last_hidden_state
            
            null_embed_batch = null_emb.to(context.device)
            null_embed_batch = null_embed_batch.expand(context.size(0), -1, -1) 

            uncond_prob = cfg.model.diffusion.uncond_prob
            if uncond_prob == 0.0: # 强制条件
                effective_context = context 
            elif uncond_prob == 1.0: # 强制无条件
                effective_context = null_embed_batch
            else: # 训练时的随机CFG
                mask = torch.rand(context.size(0),
                      device=context.device) < uncond_prob
                effective_context = torch.where(
                    mask.unsqueeze(-1).unsqueeze(-1), 
                    null_embed_batch, 
                    context) 
            mu, logvar = vae.encode(batch.node_pos, batch.bond_index)
            node_emb = vae.reparameterize(mu, logvar)

        loss_dict = diffusion.get_loss(
            node_emb=node_emb,
            edge_index=batch.bond_index,
            batch_node=batch.batch,
            num_graphs=batch.num_graphs,
            node_xy=node_xy,
            context=effective_context,
            controlnet_cond=None,
            controlnet_conditioning_scale=None,
        )
    
        loss = loss_dict["diff_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = clip_grad_norm_(model.module.parameters(), cfg.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step_each_iter:
            scheduler.step()

        # === Logging ===
        if rank == 0:
            log_msg = " | ".join([f"{k}: {v.item():.6f}" for k, v in loss_dict.items()])
            log_msg += f' | GradNorm: {grad_norm.item():.4f}'
            log_msg += f' | Lr: {optimizer.param_groups[0]["lr"]:.6f}'
            logger.info("[Train] Iter %d | %s", it_idx, log_msg)

            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), it_idx)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it_idx)
            writer.add_scalar("train/grad", grad_norm, it_idx)
            writer.flush()

        return

    def validate(it_idx: int):
        diffusion.eval()
        total_emb_mse = 0.0
        total_recon_loss = 0.0
        total_recon_loss_xy = 0.0
        num_batches = len(val_loader.dataset)

        loader = PyRenderDataLoader()
        renderer = PyRenderWrapper((512, 512))
        view_params = {
            "cam_pos_offset": np.array([0.7, 0.5, 0.7], dtype=np.float32) * 0.6, 
            "up_vector": [0, 1, 0]
        }

        for _ in range(len(val_loader.dataset)):
            batch = next(val_iter).to(device)
            context = batch.get("clip_emb", None)
            with torch.no_grad():
                if context is None:
                    text_inputs = tokenizer(batch.caption, return_tensors="pt", padding='max_length', truncation=True)
                    input_ids = text_inputs["input_ids"].to(device)
                    attention_mask = text_inputs["attention_mask"].to(device) 
                    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    context = outputs.last_hidden_state

                original_filename_stem = Path(batch.rig_path[0]).stem
                
                batch = batch.to(device)
                node_xy = batch.node_pos[:, :2]

                null_embed_batch = null_emb.to(context.device)
                null_embed_batch = null_embed_batch.expand(context.size(0), -1, -1) 

                z_sample = diffusion.sample(
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

                mu, logvar = vae.encode(batch.node_pos, batch.bond_index)
                emb_mse = F.mse_loss(mu, z_sample)
                logger.info(f"[Val] EmbMSE: {emb_mse.item():.4f}, File: {batch.rig_path[0]}")

                recon_pos = vae.decode(z_sample, batch.bond_index)
                recon_loss = F.mse_loss(recon_pos, batch.node_pos, reduction='mean')
                recon_loss_xy = F.mse_loss(recon_pos[:, :2], batch.node_pos[:, :2], reduction='mean')
                total_emb_mse += emb_mse.item()
                total_recon_loss += recon_loss.item()
                total_recon_loss_xy += recon_loss_xy.item()
                print(f"Sample {original_filename_stem} | Recon Loss (MSE): {recon_loss.item():.6f} | Recon Loss XY (MSE): {recon_loss_xy.item():.6f}")

            view = batch["view"][0]
            recon_pos = recon_pos * batch.scale + batch.translation
            if view == "yz":
                recon_pos = recon_pos[..., [2, 1, 0]]
            elif view == "xz":
                recon_pos = recon_pos[..., [0, 2, 1]]

            skeleton_save_path = Path(log_dir) / "skeleton.txt"
            save_skeleton_file(skeleton_save_path, recon_pos, batch.bond_index)

            coord_gt = batch.node_pos.cpu()
            coord_recon = recon_pos.cpu()
            hierarchy = edge_index_to_hierarchy(batch.bond_index)
            joint_dict_gt = build_joint_dict(coord_gt)
            joint_dict_recon = build_joint_dict(coord_recon)

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
            # 保存
            vis_dir = Path(log_dir) / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_dir / f"{original_filename_stem}_{it_idx}.png"
            skeleton_save_path.unlink()
            
            plt.suptitle(f"Recon MSE: {recon_loss.item():.4f}, XY MSE: {recon_loss_xy.item():.4f}\n{batch.caption[0]}", y=1.02)
            plt.tight_layout()
            plt.savefig(vis_path, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"[Vis] Saved {vis_path}")

        writer.add_scalar("val/emb_mse", total_emb_mse / num_batches, it_idx)
        writer.add_scalar("val/recon_loss", total_recon_loss / num_batches, it_idx)
        writer.add_scalar("val/recon_loss_xy", total_recon_loss_xy / num_batches, it_idx)
        writer.flush()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_it = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        if rank == 0:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        torch.distributed.barrier()
        start_it = ckpt.get("iteration", 0)
        logger.info("✅ Resumed from %s @ iter %d", args.resume, start_it)

    try:
        for it in range(start_it, cfg.train.max_iters + 1):
            train(it)
            if (it % cfg.train.val_freq == 0 and it != 0) or it == cfg.train.max_iters:
                torch.distributed.barrier()
                if rank == 0:
                    validate(it)
                torch.distributed.barrier()
            if (it % 1000 == 0 and it != 0) or it == cfg.train.max_iters:
                torch.distributed.barrier()
                if rank == 0:
                    # save checkpoint
                    torch.save({
                        "config": cfg,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "iteration": it,
                    }, Path(ckpt_dir, f"{it}.pt"))
                torch.distributed.barrier()
    except KeyboardInterrupt:
        logger.info("Interrupted by user…")
    finally:
        logger.info("Training finished or terminated.")
        if rank == 0:
            writer.close()