import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
from functools import partial
from tqdm import tqdm
from torch.amp import autocast
from einops import rearrange, reduce
from typing import Optional

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        controlnet_model: nn.Module = None,
        device = None
    ):
        super().__init__()

        self.base_model = base_model
        self.controlnet = controlnet_model

        self.channels = self.base_model.latent_dim if hasattr(self.base_model, 'latent_dim') else None
        assert self.channels is not None, 'latent model must have latent_dim'

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be pred_noise, pred_x0, or pred_v'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        if device is not None:
            betas = betas.to(device)
        elif hasattr(base_model, 'device'):
            device = base_model.device
            betas = betas.to(device)
        self._device = device

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)
        else: # Should not happen due to assert, but for safety
            loss_weight = torch.ones_like(snr)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self,
                          node_emb, 
                          t_expanded, 
                          edge_index,
                          batch_node,
                          node_xy,
                          context,
                          controlnet_cond, 
                          controlnet_conditioning_scale,
                          clip_denoised,
                        ):
        if controlnet_cond is not None and controlnet_cond.numel() > 0 :
            controlnet_cond = controlnet_cond.to(self.device)
            control_residuals = self.controlnet(
                x=node_emb,
                edge_index=edge_index,
                t=t_expanded,
                batch_node=batch_node,
                node_xy=node_xy,
                context=context,
                controlnet_cond=controlnet_cond,
                conditioning_scale=controlnet_conditioning_scale,
            )
        else:
            control_residuals = None

        base_model_output = self.base_model(
            x=node_emb,
            edge_index=edge_index,
            t=t_expanded,
            batch_node=batch_node,
            node_xy=node_xy,
            context=context,
            controlnet_residuals=control_residuals
        )
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_denoised else identity

        if self.objective == 'pred_noise':
            pred_noise = base_model_output
            x_start = self.predict_start_from_noise(
                x_t=node_emb,
                t=t_expanded,
                noise=pred_noise
            )
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = base_model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(
                x_t=node_emb,
                t=t_expanded,
                x0=x_start
            )
        elif self.objective == 'pred_v':
            v = base_model_output
            x_start = self.predict_start_from_v(
                x_t=node_emb,
                t=t_expanded,
                v=v
            )
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(
                x_t=node_emb,
                t=t_expanded,
                x0=x_start
            )
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def sample(
        self,
        num_nodes,
        edge_index,
        batch_node,
        num_graphs,
        node_xy,
        context,
        null_text_emb,
        cfg_scale_text,
        controlnet_cond,
        controlnet_conditioning_scale,
        cfg_scale_control,
        clip_denoised,
    ):
        self.eval()
        if not self.channels:
             raise ValueError("Model latent_dim not available...")
        device = self.betas.device

        node_emb = torch.randn([num_nodes, self.channels], device=device)

        if controlnet_cond is not None:
            null_cond = torch.zeros_like(controlnet_cond).to(device)
        else:
            null_cond = None
            cfg_scale_control = 1

        for t_int in tqdm(reversed(range(0, self.num_timesteps)), desc='DDPM sampling loop', total=self.num_timesteps):
            time_cond_scalar = torch.full((num_graphs,), t_int, device=device, dtype=torch.long)
            t_expanded = time_cond_scalar.index_select(0, batch_node)

            preds_text_control = self.model_predictions(
                node_emb=node_emb,
                t_expanded=t_expanded,
                edge_index=edge_index,
                batch_node=batch_node,
                node_xy=node_xy,
                context=context,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                clip_denoised=False,
            )
            eps_text_control = preds_text_control.pred_noise

            preds_untext_control = self.model_predictions(
                node_emb=node_emb,
                t_expanded=t_expanded,
                edge_index=edge_index,
                batch_node=batch_node,
                node_xy=node_xy,
                context=null_text_emb,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                clip_denoised=False,
            )
            eps_untext_control = preds_untext_control.pred_noise

            # print(null_cond, controlnet_cond, cfg_scale_text)
            # print(F.mse_loss(eps_text_uncontrol, eps_text_control), F.mse_loss(eps_untext_uncontrol, eps_untext_control))
            # 暂时先这样
            combined_eps = eps_untext_control + cfg_scale_text * (eps_text_control - eps_untext_control)

            x_start = self.predict_start_from_noise(
                x_t=node_emb,
                t=t_expanded,
                noise=combined_eps,
            )

            if clip_denoised:
                 x_start.clamp_(-1., 1.)

            model_mean, posterior_variance, model_log_variance = self.q_posterior(
                x_start=x_start,
                x_t=node_emb, 
                t=t_expanded
            )
            noise = torch.randn_like(node_emb) if t_int > 0 else 0.
            node_emb = model_mean + (0.5 * model_log_variance).exp() * noise

        self.train()
        return node_emb

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self,
                 x_start,
                 edge_index,
                 t_expanded,
                 batch_node,
                 node_xy,
                 context,
                 controlnet_cond,
                 controlnet_conditioning_scale,
                ):

        noise = torch.randn_like(x_start)
        x = self.q_sample(x_start=x_start, t=t_expanded, noise=noise)

        if controlnet_cond is not None and controlnet_cond.numel() > 0 :
            control_residuals = self.controlnet(
                x=x,
                edge_index=edge_index,
                t=t_expanded,
                node_xy=node_xy,
                batch_node=batch_node,
                context=context,
                controlnet_cond=controlnet_cond,
                conditioning_scale=controlnet_conditioning_scale
            )
        else:
            control_residuals = None

        base_model_output = self.base_model(
            x=x,
            edge_index=edge_index,
            t=t_expanded,
            batch_node=batch_node,
            node_xy=node_xy,
            context=context,
            controlnet_residuals=control_residuals,
        )

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t_expanded, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(base_model_output, target, reduction='none')

        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t_expanded, loss.shape)

        return loss.mean()

    def forward(self, 
                node_emb, 
                edge_index, 
                time_step,
                batch_node,
                node_xy,
                context,
                controlnet_cond,
                controlnet_conditioning_scale,
                ):
        node_emb = node_emb.to(self.device)
        edge_index = edge_index.to(self.device)
        time_step = time_step.to(self.device)

        if context is not None:
            context = context.to(self.device)
        if controlnet_cond is not None:
            controlnet_cond = controlnet_cond.to(self.device)

        return self.p_losses(
            x_start=node_emb,
            edge_index=edge_index,
            t_expanded=time_step,
            batch_node=batch_node,
            node_xy=node_xy,
            context=context,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

    def sample_time(self, num_graphs, device):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        return time_step

    def get_loss(self,
                 node_emb,
                 edge_index,
                 batch_node,
                 num_graphs,
                 node_xy,
                 context,
                 controlnet_cond,
                 controlnet_conditioning_scale,
                ):
        device = node_emb.device

        assert batch_node.max().item() < num_graphs, "batch_node max value exceeds num_graphs"

        time_step = self.sample_time(num_graphs, device)
        t_expanded = time_step.index_select(0, batch_node)

        loss = self(
            node_emb=node_emb,
            edge_index=edge_index,
            time_step=t_expanded,
            batch_node=batch_node,
            node_xy=node_xy,
            context=context,
            controlnet_cond=controlnet_cond,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        loss_dict = {
            'diff_loss': loss
        }
        return loss_dict