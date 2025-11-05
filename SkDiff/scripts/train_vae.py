import sys
import os
import shutil
import argparse
from pathlib import Path
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

from utils.misc import load_config, seed_all, get_new_log_dir, get_logger
from utils.transform import FeaturizeGraph
from utils.dataset import build_dataloader
from utils.train import get_optimizer, get_scheduler, inf_iterator, should_step_each_iter
from models.vae import NodeCoordVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_vae.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='logs')
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

    # === Model ===
    logger.info('Building model...')
    if cfg.model.name == 'vae':
        model = NodeCoordVAE(
            cfg.model.coord_dim, # N*3
            cfg.model.hidden_dim, # 128
            cfg.model.latent_dim, # 32
            cfg.model.norm_type # layer
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % cfg.model.name)
    print('Num of trainable parameters is', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    # === Optim & sched ===
    optimizer = get_optimizer(cfg.train.optimizer, model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    step_each_iter = should_step_each_iter(cfg.train.scheduler.type)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.train.use_amp)
    
    def train_step(it_idx: int):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch = next(train_iter).to(args.device)
        x = batch.node_pos
        edge_index = batch.bond_index

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.use_amp):
            
            x_recon, mu, logvar = model(x, edge_index)
            loss, recon_loss, kl_loss = model.get_loss(x, x_recon, mu, logvar, beta=cfg.train.kl_beta)
        
        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update() 

        if step_each_iter:
            scheduler.step()
    
        log_msg = " | ".join([f"{k}: {v.item():.6f}" for k, v in loss_dict.items()])
        logger.info("[Train] Iter %d | %s", it_idx, log_msg)
        for k, v in loss_dict.items():
            writer.add_scalar(f"train/{k}", v.item(), it_idx)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it_idx)
        writer.add_scalar("train/grad", grad_norm, it_idx)
        writer.flush()


    def validate(it):
        sum_n =  0   # num of loss
        sum_loss_dict = {} 
        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                x = batch.node_pos
                edge_index = batch.bond_index

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.use_amp):
                    x_recon, mu, logvar = model(x, edge_index)
                    loss, recon_loss, kl_loss = model.get_loss(x, x_recon, mu, logvar, beta=cfg.train.kl_beta)
                
                loss_dict = {
                    "loss": loss,
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss
                }
                if len(sum_loss_dict) == 0:
                    sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                else:
                    for key in sum_loss_dict.keys():
                        sum_loss_dict[key] += loss_dict[key].item()
                sum_n += 1

        avg_loss_dict = {k: v / sum_n for k, v in sum_loss_dict.items()}
        avg_loss = avg_loss_dict['loss']
        # update lr scheduler
        if cfg.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif cfg.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        log_info = '[Validate] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items()
        ])
        logger.info(log_info)
        for k, v in avg_loss_dict.items():
            writer.add_scalar('val/%s' % k, v, it)
        writer.flush()
        return avg_loss
    
    try:
        model.train()
        for it in range(1, cfg.train.max_iters+1):
            try:
                train_step(it)
            except RuntimeError as e:
                logger.error('Runtime Error ' + str(e))
                logger.error('Skipping Iteration %d' % it)
            if it % cfg.train.val_freq == 0 or it == cfg.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': cfg,
                    'encoder': model.encoder.state_dict(),
                    'decoder': model.decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)

                model.train()
    except KeyboardInterrupt:
        logger.info('Keyboard Interrupt')
    finally:
        logger.info('Saving final checkpoint...')
                    