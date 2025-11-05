import copy
import warnings
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, CosineAnnealingLR
import math

from .warmup import GradualWarmupScheduler

# ============================
# Learning‑rate schedulers
# ============================

class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    """Standard ExponentialLR but with a floor on the LR value."""

    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, gamma, last_epoch, verbose)

    # Override both step‑based & closed‑form interfaces
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group["lr"] * self.gamma, self.min_lr) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr) for base_lr in self.base_lrs]


# ----------------------------------------------------
# Convenience helpers
# ----------------------------------------------------

def repeat_data(data: Data, num_repeat):
    datas = [copy.deepcopy(data) for _ in range(num_repeat)]
    return Batch.from_data_list(datas)

def repeat_batch(batch: Batch, num_repeat):
    datas = batch.to_data_list()
    new_data = []
    for _ in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)

def inf_iterator(iterable, distributed=False):
    epoch = 0
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            epoch += 1
            if distributed:
                iterable.sampler.set_epoch(epoch)

# ----------------------------------------------------
# Optimizer builder
# ----------------------------------------------------

def get_optimizer(cfg, model):
    if cfg.type.lower() == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                betas=(cfg.beta1, cfg.beta2))
    elif cfg.type.lower() == "adamw":
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay,
                                 betas=(cfg.beta1, cfg.beta2))
    else:
        raise NotImplementedError(f"Optimizer not supported: {cfg.type}")


# ----------------------------------------------------
# Scheduler builder (NEW: MultiStepLR & Warm‑up version)
# ----------------------------------------------------

def get_scheduler(cfg, optimizer):
    ctype = cfg.type.lower()

    # -------- Plateau families --------
    if ctype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor,
                                                          patience=cfg.patience, min_lr=cfg.min_lr)
    if ctype == "warmup_plateau":
        after = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor,
                                                           patience=cfg.patience, min_lr=cfg.min_lr)
        return GradualWarmupScheduler(optimizer, multiplier=cfg.multiplier,
                                      total_epoch=cfg.total_epoch, after_scheduler=after)

    # -------- Exponential families --------
    if ctype == "expmin":
        return ExponentialLR_with_minLr(optimizer, gamma=cfg.factor, min_lr=cfg.min_lr)
    if ctype == "expmin_milestone":
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(optimizer, gamma=gamma, min_lr=cfg.min_lr)

    # -------- Cosine (iter‑wise scheduling) --------
    if ctype == "cosine":
        warmup = getattr(cfg, "warmup", 10000)
        total = getattr(cfg, "total_iters", 1000000)
        min_lr = getattr(cfg, "min_lr", 1e-6)
        base_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(it):
            if it < warmup:
                return it / warmup
            prog = (it - warmup) / max(1, total - warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * prog))
            return cosine_decay * (1 - min_lr / base_lr) + min_lr / base_lr

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------- NEW: PyTorch CosineAnnealingLR Schedulers --------
    if ctype in {"cosineannealing", "cosineannealinglr"}:
        return CosineAnnealingLR(optimizer,
                                 T_max=cfg.T_max,
                                 eta_min=getattr(cfg, 'eta_min', 0))

    if ctype in {"warmup_cosineannealing", "warmup_cosineannealinglr"}:
        print(cfg)
    # 检查YAML中是否手动指定了 T_max，以启用“快速衰减”模式
        if hasattr(cfg, 'T_max') and cfg.T_max is not None:
            t_max_value = cfg.T_max
            print(f"Using user-defined T_max for aggressive decay: {t_max_value}")
        else:
            # 如果未指定T_max，则自动计算，启用“全程衰减”模式
            t_max_value = cfg.total_iters - cfg.warmup_iter
            if t_max_value <= 0:
                raise ValueError("With auto T_max, total_iters must be greater than warmup_iter.")
            print(f"Calculated T_max for full-length decay: {t_max_value}")

        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max_value, # 使用我们最终决定的 t_max_value
            eta_min=getattr(cfg, 'eta_min', 1e-6)
        )
        return GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=cfg.warmup_iter,
            after_scheduler=main_scheduler
        )
    # -------- NEW: MultiStepLR --------
    if ctype in {"multisteplr", "multistep"}:
        return MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    # -------- NEW: Warm‑up + MultiStepLR --------
    if ctype in {"warmup_multistep", "warmup_multisteplr"}:
        warmup_iters = getattr(cfg, "warmup_iter", 10000)
        multistep = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        return GradualWarmupScheduler(optimizer, multiplier=1.0,
                                      total_epoch=warmup_iters, after_scheduler=multistep)
    if ctype == "cosine_hold":
        warmup_iters = getattr(cfg, 'warmup_iter', 0)
        decay_iters = cfg.T_max  # 使用 T_max 作为衰减的步数
        eta_min = getattr(cfg, 'eta_min', 0)
        
        # 衰减结束的迭代点
        decay_end_iter = warmup_iters + decay_iters
        
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_ratio = eta_min / base_lr
        
        def lr_lambda(current_step: int):
            # 阶段1: 预热 (Warmup)
            if current_step < warmup_iters:
                return float(current_step) / float(max(1, warmup_iters))
            
            # 阶段2: 余弦衰减 (Cosine Decay)
            elif current_step < decay_end_iter:
                # 计算衰减阶段的进度 (0 to 1)
                progress = float(current_step - warmup_iters) / float(max(1, decay_iters))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            
            # 阶段3: 保持在最低学习率 (Hold)
            else:
                return min_lr_ratio

        return LambdaLR(optimizer, lr_lambda)
    
    raise NotImplementedError(f"Scheduler not supported: {cfg.type}")

def should_step_each_iter(scheduler_type: str) -> bool:
    """Return True if scheduler.step() should be called every training iteration."""
    t = scheduler_type.lower()
    return t not in {"plateau", "warmup_plateau"}
