import sys
import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
# for import other sub folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
import numpy as np
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from transformers import CLIPTokenizer, CLIPTextModel

from utils.misc import load_config, seed_all, get_new_log_dir, get_logger
from utils.transform import FeaturizeGraph
from utils.dataset import build_dataloader
from utils.train import get_optimizer, get_scheduler, inf_iterator, should_step_each_iter
from models.graph_model import GraphLatentModel
from models.controlnet import ControlNetGraphModel
from models.diffusion import GaussianDiffusion
from models.vae import NodeCoordVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/train_controlnet.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="path to ControlNet checkpoint (.pt) to resume training")
    args = parser.parse_args()

    # === Load YAML ===
    cfg = load_config(args.config)
    cfg_name = Path(args.config).stem

    # === Seed & log dir ===
    seed_all(cfg.train.seed)
    log_dir = get_new_log_dir(args.logdir, prefix=cfg_name)
    ckpt_dir = Path(log_dir, "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    shutil.copyfile(args.config, Path(log_dir, Path(args.config).name))

    logger.info("Using config %s", args.config)
    logger.info(f"Logging to: {log_dir}")
    logger.info(f"Using device: {args.device}")

    # === Data ===
    featurizer = FeaturizeGraph(use_rotate=cfg.dataset.get('use_rotate', True))
    train_loader = build_dataloader(cfg, "train", featurizer)
    val_loader = build_dataloader(cfg, "val", featurizer)
    train_iter = inf_iterator(train_loader)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # === CLIP Loading (Freeze) ===
    logger.info("Initializing CLIP model for captions...")
    clip_ckpt_path = cfg.model.clip.get('ckpt_path', None)
    if not clip_ckpt_path:
         raise FileNotFoundError(f"CLIP checkpoint path not found or specified: {clip_ckpt_path}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt_path)
    text_encoder = CLIPTextModel.from_pretrained(clip_ckpt_path).to(args.device)

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    logger.info("CLIP loaded and frozen.")
    empty_ids = tokenizer("", return_tensors="pt").input_ids.to(args.device)
    null_emb  = text_encoder(empty_ids).pooler_output.squeeze(0)

    # === VAE Loading (Freeze) ===
    logger.info("Loading and freezing VAE...")
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
    logger.info(f"Loading VAE weights from: {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=args.device, weights_only=False)
    # Load VAE state dict robustly
    vae_state_dict = vae_ckpt.get('model_state_dict', vae_ckpt.get('model', vae_ckpt)) # Try common keys
    if vae_state_dict:
        # Handle potential nested structure if VAE was saved within a larger dict
        if 'encoder' in vae_state_dict and 'decoder' in vae_state_dict:
             vae.encoder.load_state_dict(vae_state_dict['encoder'])
             vae.decoder.load_state_dict(vae_state_dict['decoder'])
        else: # Assume it's the full VAE model state_dict
             vae.load_state_dict(vae_state_dict)
        logger.info("VAE weights loaded.")
    else:
        raise KeyError(f"Could not find VAE state dict in checkpoint: {vae_ckpt_path}")

    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    logger.info("VAE loaded and frozen.")

    # === Base Model ===
    logger.info("Building models...")
    base_model = GraphLatentModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout,
        heads=cfg.model.diffusion.head,
    ).to(args.device)

    # --- Load Pre-trained Base Model Weights (Mandatory) ---
    base_ckpt_path = cfg.model.diffusion.get('ckpt_path', None)
    if base_ckpt_path and os.path.isfile(base_ckpt_path):
        logger.info(f"Loading pre-trained base model weights from: {base_ckpt_path}")
        base_ckpt = torch.load(base_ckpt_path, map_location=args.device, weights_only=False)
        base_model_state_dict = base_ckpt.get('model', base_ckpt.get('ema', base_ckpt))
        if base_model_state_dict is None:
             raise KeyError(f"Could not find model state dict in base checkpoint: {base_ckpt_path}")
        # Filter state dict to match base_model keys EXACTLY before loading strictly
        base_model_keys = set(base_model.state_dict().keys())
        filtered_base_state_dict = {k: v for k, v in base_model_state_dict.items() if k in base_model_keys}
        missing_keys_base, unexpected_keys_base = base_model.load_state_dict(filtered_base_state_dict, strict=False) # Load non-strictly first to see differences
        if unexpected_keys_base:
            logger.warning(f"Unexpected keys found when loading base model state_dict (will be ignored): {unexpected_keys_base}")
        if missing_keys_base:
            logger.error(f"Missing keys when loading base model state_dict: {missing_keys_base}. Check checkpoint compatibility.")
        else:
             logger.info(f"Base model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Base model checkpoint path is required but not found or specified: {base_ckpt_path}")

    # === Weight Loading ===
    start_it = 0
    optimizer_state = None # Initialize optimizer/scheduler states
    scheduler_state = None
    ema_state = None
    # --- Load/Resume ControlNet (Optional) ---
    controlnet_model = ControlNetGraphModel(
        latent_dim=cfg.model.diffusion.latent_dim,
        hidden_dim=cfg.model.diffusion.hidden_dim,
        depth=cfg.model.diffusion.depth,
        conditioning_channels=cfg.model.controlnet.conditioning_channels,
        clip_embed_dim=cfg.model.clip.clip_embed_dim,
        dropout=cfg.model.diffusion.dropout, # Usually match dropout
        heads=cfg.model.diffusion.head,
    ).to(args.device)

    controlnet_ckpt_path = cfg.model.controlnet.get('ckpt_path', None)
    controlnet_loaded_from_ckpt = False
    if args.resume:
         controlnet_ckpt_path = args.resume
         logger.info(f"Attempting to resume ControlNet training from explicit checkpoint: {controlnet_ckpt_path}")

    if controlnet_ckpt_path and os.path.isfile(controlnet_ckpt_path):
        logger.info(f"Loading ControlNet state from checkpoint: {controlnet_ckpt_path}")
        ctrl_ckpt = torch.load(controlnet_ckpt_path, map_location=args.device, weights_only=False)

        # Load ControlNet model weights
        ctrl_model_state_dict = ctrl_ckpt.get('controlnet_model', ctrl_ckpt.get('model'))
        if ctrl_model_state_dict is None:
             logger.error(f"Could not find ControlNet model state dict in checkpoint: {controlnet_ckpt_path}. Will train ControlNet from scratch/copied weights.")
        else:
            try:
                load_result_ctrl = controlnet_model.load_state_dict(ctrl_model_state_dict, strict=True)
                logger.info(f"ControlNet model weights loaded successfully: {load_result_ctrl}")
                controlnet_loaded_from_ckpt = True # Mark as loaded

                # Load optimizer, scheduler, EMA, iteration only if successfully loaded model weights
                optimizer_state = ctrl_ckpt.get('optimizer')
                scheduler_state = ctrl_ckpt.get('scheduler')
                ema_state = ctrl_ckpt.get('ema')
                start_it = ctrl_ckpt.get('iteration', 0) + 1
                logger.info(f"Resuming training state from iteration {start_it}")
                if not optimizer_state: logger.warning("Optimizer state not found in resume checkpoint.")
                if not scheduler_state: logger.warning("Scheduler state not found in resume checkpoint.")
                if not ema_state: logger.warning("EMA state not found in resume checkpoint.")

            except RuntimeError as e:
                 logger.error(f"Error loading ControlNet state_dict (likely model structure mismatch): {e}. Will train ControlNet from scratch/copied weights.")
                 start_it = 0 # Reset iteration if loading failed

    # --- Initsialize ControlNet by Copying if Not Loaded from Checkpoint (Optimized) ---
    if not controlnet_loaded_from_ckpt:
        logger.info("ControlNet weights not loaded from checkpoint. Attempting to initialize internal layers by copying from base model...")
        base_sd = base_model.state_dict()
        ctrl_sd = controlnet_model.state_dict()
        copied_count = 0
        skipped_non_copyable = 0 # Count layers skipped because they are ControlNet specific
        skipped_missing = 0      # Count layers skipped because not found in base or shape mismatch
        copied_keys = set()      # Keep track of keys successfully copied

        # Get the state dict for update
        updated_ctrl_sd = ctrl_sd.copy()

        # Define prefixes or exact names of layers that should NOT be copied (Zero Initialized)
        # Adjust these based on your ControlNetGraphModel naming convention
        non_copyable_prefixes = ("controlnet_blocks.", "controlnet_cond_embedding.")

        # Iterate through ControlNet state dict keys
        for key in ctrl_sd:
            # Check if the layer should be skipped (ControlNet specific zero-init layers)
            is_non_copyable = False
            for prefix in non_copyable_prefixes:
                if key.startswith(prefix):
                    is_non_copyable = True
                    skipped_non_copyable += 1
                    # logger.debug(f"Skipping non-copyable layer: {key}") # Optional debug log
                    break # No need to check other prefixes for this key

            if is_non_copyable:
                continue # Move to the next key if it's a non-copyable layer

            # Attempt to copy from base model if it's not a non-copyable layer
            if key in base_sd:
                if ctrl_sd[key].shape == base_sd[key].shape:
                    updated_ctrl_sd[key] = base_sd[key].clone()
                    copied_count += 1
                    copied_keys.add(key)
                    # logger.debug(f"Copied layer: {key}") # Optional debug log
                else:
                    logger.warning(f"Shape mismatch, skipping copy for layer '{key}': Base shape {base_sd[key].shape}, ControlNet shape {ctrl_sd[key].shape}")
                    skipped_missing += 1
            else:
                # Layer exists in ControlNet but not in Base Model (or name differs)
                # Keep the original ControlNet initialization for this layer
                logger.warning(f"Layer '{key}' not found in base_model state_dict. Keeping ControlNet's default initialization.")
                skipped_missing += 1

        # Load the updated state dict into ControlNet model
        try:
            # Use strict=False initially to report unexpected keys if any (shouldn't happen here)
            # Then consider strict=True if confident
            missing_keys_ctrl, unexpected_keys_ctrl = controlnet_model.load_state_dict(updated_ctrl_sd, strict=False)
            if unexpected_keys_ctrl:
                 # This shouldn't happen if we started from ctrl_sd.copy()
                 logger.error(f"Unexpected keys found after copying weights (ERROR in logic?): {unexpected_keys_ctrl}")
            if missing_keys_ctrl:
                 # This also shouldn't happen if we started from ctrl_sd.copy()
                 logger.error(f"Missing keys found after copying weights (ERROR in logic?): {missing_keys_ctrl}")

            logger.info(f"Initialization by copying complete: Copied {copied_count} parameter tensor(s).")
            logger.info(f"Skipped {skipped_non_copyable} non-copyable ControlNet-specific parameter tensor(s).")
            logger.info(f"Skipped {skipped_missing} parameter tensor(s) due to missing key in base or shape mismatch.")


        except Exception as e:
            logger.error(f"Error loading state_dict after attempting to copy weights: {e}. ControlNet initialization might be incomplete.")

        # *** DO NOT COPY THESE (Zero Initialized) ***
        # 'controlnet_cond_embedding.weight', 'controlnet_cond_embedding.bias'
        # f'controlnet_blocks.{i}.weight', f'controlnet_blocks.{i}.bias'

    # === Freeze Base Model ===
    logger.info("Freezing base model parameters and setting to eval mode...")
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    # === Ensure ControlNet parameters are trainable ===
    logger.info("Ensuring ControlNet parameters are trainable...")
    trainable_params = 0
    total_params_ctrl = 0
    for name, param in controlnet_model.named_parameters():
        if not param.requires_grad: # Check if any were frozen accidentally
             logger.warning(f"ControlNet parameter '{name}' was not set to requires_grad=True. Forcing it.")
             param.requires_grad = True
        trainable_params += param.numel()
        total_params_ctrl += param.numel()
    logger.info(f'Total parameters in ControlNet: {total_params_ctrl}')
    logger.info(f'Trainable parameters (ControlNet): {trainable_params}')

    

    # === Diffusion Wrapper ===
    logger.info("Initializing Gaussian Diffusion Wrapper...")
    diffusion = GaussianDiffusion(
        base_model=base_model,
        timesteps=cfg.model.diffusion.num_steps,
        sampling_timesteps=cfg.model.diffusion.num_steps,
        objective=cfg.model.diffusion.objective,
        beta_schedule="cosine",
        min_snr_loss_weight=cfg.model.diffusion.min_snr_loss_weight,
        min_snr_gamma=cfg.model.diffusion.min_snr_gamma,
        controlnet_model=controlnet_model,
        device=args.device,
    )

    optimizer = get_optimizer(cfg.train.optimizer, controlnet_model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    step_each_iter = should_step_each_iter(cfg.train.scheduler.type)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)

    # === EMA (Targets only ControlNet parameters) ===
    ema = ExponentialMovingAverage(
        filter(lambda p: p.requires_grad, controlnet_model.parameters()),
        decay=cfg.train.get('ema_decay', 0.9999)
    )

    # === Load Optimizer, Scheduler, EMA states if resuming ===
    if controlnet_loaded_from_ckpt:
        try:
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                logger.info("Successfully loaded optimizer state.")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
                logger.info("Successfully loaded scheduler state.")
            if ema_state:
                ema.load_state_dict(ema_state)
                logger.info("Successfully loaded EMA state.")
        except Exception as e:
            logger.error(f"Error loading optimizer/scheduler/EMA state from resume checkpoint: {e}.")

    # === Training Loop ===
    def train(it: int):
        diffusion.train()
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iter).to(args.device)

        x_gt_pos = batch.node_pos # Ground truth positions
        control_conditioning_channels = cfg.model.controlnet.conditioning_channels
        control_signal = x_gt_pos[:, :control_conditioning_channels].clone().detach()

        with torch.no_grad():
            text_inputs = tokenizer(batch.captions, return_tensors="pt", padding='max_length', truncation=True)
            input_ids = text_inputs["input_ids"].to(args.device)
            attention_mask = text_inputs["attention_mask"].to(args.device) 
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            caption_emb = outputs.pooler_output

            null_embed_batch = null_emb.to(caption_emb.device)  
            null_embed_batch = null_embed_batch.unsqueeze(0).expand(caption_emb.size(0), -1) 

            uncond_prob=cfg.model.diffusion.uncond_prob
            if uncond_prob == 0.0: # 强制条件
                effective_caption_emb = caption_emb 
            elif uncond_prob == 1.0: # 强制无条件
                effective_caption_emb = null_embed_batch
            else: # 训练时的随机CFG
                mask = torch.rand(caption_emb.size(0),
                      device=caption_emb.device) < uncond_prob  # (B,)
                effective_caption_emb = torch.where(
                    mask.unsqueeze(-1),      # (B, 1)
                    null_embed_batch,        # 无条件
                    caption_emb)             # 有条件
            mu, logvar = vae.encode(x_gt_pos, batch.bond_index)
            node_emb = vae.reparameterize(mu, logvar)

        loss_dict = diffusion.get_loss(
            node_emb=node_emb,
            edge_index=batch.bond_index,
            batch_node=batch.batch,
            num_graphs=batch.num_graphs,
            caption_emb=effective_caption_emb,
            controlnet_cond=control_signal,
            controlnet_conditioning_scale=1.0,
        )
    
        loss = loss_dict["diff_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable_params = filter(lambda p: p.requires_grad, controlnet_model.parameters())
        orig_grad_norm = clip_grad_norm_(trainable_params, cfg.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step_each_iter:
            scheduler.step()

        ema.update()

        # === Logging ===
        log_items = {k: v.item() for k, v in loss_dict.items() if torch.is_tensor(v)}
        log_info = '[Train] Iter %d | Loss: %.6f | ' % (it, loss.item()) + ' | '.join([
            '%s: %.6f' % (k, v) for k, v in log_items.items() if k != 'diff_loss' # Avoid duplicate loss log
        ])
        log_info += f' | GradNorm: {orig_grad_norm.item():.4f}'
        logger.info(log_info)
        for k, v in log_items.items():
            writer.add_scalar(f'train/{k}', v, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm.item(), it)
        writer.add_scalar('train/amp_scale', scaler.get_scale(), it)
        writer.flush()

    def validate(it: int):
        with ema.average_parameters():
            diffusion.eval()
        sum_n = 0
        sum_dict = defaultdict(float)
        sum_recon_loss = 0.0
        sum_recon_xy_loss = 0.0


        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                batch = batch.to(args.device)
                x_gt_pos = batch.node_pos # Ground truth positions
                control_conditioning_channels = cfg.model.controlnet.conditioning_channels
                control_signal = x_gt_pos[:, :control_conditioning_channels].clone().detach()
                
                text_inputs = tokenizer(batch.captions, return_tensors="pt", padding='max_length', truncation=True)
                input_ids = text_inputs["input_ids"].to(args.device)
                attention_mask = text_inputs["attention_mask"].to(args.device) 
                outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                caption_emb = outputs.pooler_output

                null_embed_batch = null_emb.to(caption_emb.device)
                null_embed_batch = null_embed_batch.unsqueeze(0).expand(caption_emb.size(0), -1) 

                mu, logvar = vae.encode(batch.node_pos, batch.bond_index)
                node_emb = vae.reparameterize(mu, logvar)

                loss_cond = diffusion.get_loss(
                    node_emb=node_emb,
                    edge_index=batch.bond_index,
                    batch_node=batch.batch,
                    num_graphs=batch.num_graphs,
                    caption_emb=caption_emb,
                    controlnet_cond=control_signal,
                    controlnet_conditioning_scale=1.0,
                )
                loss_uncond = diffusion.get_loss(
                    node_emb=node_emb,
                    edge_index=batch.bond_index,
                    batch_node=batch.batch,
                    num_graphs=batch.num_graphs,
                    caption_emb=null_embed_batch,
                    controlnet_cond=control_signal,
                    controlnet_conditioning_scale=1.0,
                )
                # accumulate
                for k in loss_cond: sum_dict[f"{k}_cond"] += loss_cond[k].item()
                for k in loss_uncond: sum_dict[f"{k}_uncond"] += loss_uncond[k].item()
                

                z_sample_batch = diffusion.sample(
                    num_nodes=batch.num_nodes,
                    edge_index=batch.bond_index,
                    batch_node=batch.batch,
                    num_graphs=batch.num_graphs,
                    caption_emb=caption_emb,
                    null_text_emb=null_embed_batch,
                    cfg_scale_text=cfg.model.diffusion.cfg_scale,
                    controlnet_cond=control_signal,
                    controlnet_conditioning_scale=1.0,
                    cfg_scale_control=1,
                    clip_denoised=True,
                    )

                recon_pos = vae.decode(z_sample_batch, batch.bond_index)
                
                recon_loss = F.mse_loss(recon_pos, x_gt_pos, reduction='mean')
                sum_recon_loss += recon_loss.item()
                
                recon_loss_xy = F.mse_loss(recon_pos[:, :2], x_gt_pos[:, :2], reduction='mean')
                sum_recon_xy_loss += recon_loss_xy.item()
                sum_n += 1
        avg_loss_dict = {k: v / sum_n for k, v in sum_dict.items()}
        avg_loss_dict['recon_mse'] = sum_recon_loss / sum_n
        avg_loss_dict['recon_mse_xy'] = sum_recon_xy_loss / sum_n
        avg_diffusion_loss = avg_loss_dict.get('diff_loss', float('inf'))

        # === Step LR Scheduler based on validation loss ===
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging
        if cfg.train.scheduler.type == 'plateau':
            scheduler.step(avg_diffusion_loss)
        elif not step_each_iter: # Step if not stepped each iter (e.g., stepLR)
            scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
             logger.info(f"LR scheduler stepped. New LR: {new_lr:.6e}")


        # === Logging ===
        log_info = '[Validate] Iter %d | AvgLoss: %.6f | ' % (it, avg_diffusion_loss) + ' | '.join([
            '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items() if k != 'diff_loss'
        ])
        logger.info(log_info)
        for k, v in avg_loss_dict.items():
            writer.add_scalar(f'val/{k}', v, it)
        writer.add_scalar('val/lr', new_lr, it) # Log LR after potential scheduler step
        writer.flush()

        return avg_diffusion_loss
    
    try:
        for it in range(start_it, cfg.train.max_iters + 1):
            train(it)

            if it > 0 and (it % cfg.train.val_freq == 0 or it == cfg.train.max_iters):
                validate(it)
                # Save checkpoint for ControlNet
                ckpt_path = os.path.join(ckpt_dir, f'controlnet_{it}.pt')
                save_dict = {
                    'config': cfg,
                    'controlnet_model': controlnet_model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }
                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved ControlNet checkpoint to {ckpt_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user at iteration %d.", it)
        # Optionally save final checkpoint on interrupt
        ckpt_path = os.path.join(ckpt_dir, f'controlnet_interrupted_{it}.pt')
        save_dict = {
            'config': cfg,
            'controlnet_model': controlnet_model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
        }
        torch.save(save_dict, ckpt_path)
        logger.info(f"Saved final interrupted checkpoint to {ckpt_path}")

    except Exception as e:
         logger.error(f"An error occurred during training at iteration {it}:")
         logger.exception(e) # Log the full traceback
         # Optionally save checkpoint on other errors
         # ... save logic ...

    finally:
         writer.close()
         logger.info("Training finished or terminated.")