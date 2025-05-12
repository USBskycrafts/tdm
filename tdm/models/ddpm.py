import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from typing import Dict, List, Optional

from torch import Tensor
from tdm.modules.unet import UNet
import pytorch_optimizer as optim

class DDPM(pl.LightningModule):
    beta: Tensor
    alpha_bar: Tensor

    def __init__(
        self,
        unet_config: None | Dict = None,
        timesteps=1000,
        beta_schedule='linear',
        learning_rate=1e-4,
        cond_model='resnet34'
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet
        default_unet_config = {
            'base_ch': 64,
            'ch_scales': [1, 2, 4, 8],
            'num_res_blocks': 2,
            'attn_scales': [False, False, True, False],
            'cond_dim': 512,
            'dropout': 0.1
        }
        unet_config = unet_config or default_unet_config
        self.unet = UNet(**{**default_unet_config, **(unet_config or {})})

        # Condition model
        self.cond_model = getattr(
            torchvision.models, cond_model)(pretrained=True)
        if unet_config['in_ch'] == 1:
            conv_weight = self.cond_model.conv1.weight
            conv_weight = torch.mean(conv_weight, dim=1, keepdim=True)
            self.cond_model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cond_model.conv1.weight = nn.Parameter(conv_weight)
        self.cond_model = nn.Sequential(*list(self.cond_model.children())[:-2])

        # Diffusion parameters
        if beta_schedule == 'linear':
            beta = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            beta = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")

        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', alpha_bar)
        self.timesteps = timesteps

        self.lr = learning_rate

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward(self, x, t, aim, cond):
        # Extract condition features
        cond_features = self.cond_model(cond)

        # Add noise
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        x_noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        # Predict noise
        pred_noise = self.unet(x_noisy, t, aim, cond_features)
        return pred_noise, noise

    def training_step(self, batch, batch_idx):
        # Assume dataset returns (image, aim, feature)
        x, aim, cond, *_ = batch
        x = x.to(self.device)
        aim = aim.to(self.device)
        cond = cond.to(self.device)
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        pred_noise, true_noise = self(x, t, aim, cond)
        loss = F.mse_loss(pred_noise, true_noise)
        self.log('train/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, aim, cond, *_ = batch
        x = x.to(self.device)
        aim = aim.to(self.device)
        cond = cond.to(self.device)
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        pred_noise, true_noise = self(x, t, aim, cond)
        loss = F.mse_loss(pred_noise, true_noise)
        self.log('val/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, aim, cond, *_ = batch
        y = self._sample(x.shape, aim, cond)
        return y

    def configure_optimizers(self):
        return optim.Lion(
            self.parameters(),
            lr=self.lr
        )

    def _sample(self, shape, aim, cond=None, num_steps=None):
        device = self.device
        num_steps = num_steps or self.timesteps

        x = torch.randn(shape, device=device)
        if cond is not None:
            cond_features = self.cond_model(cond)
        else:
            raise ValueError(
                "Conditioning features must be provided for sampling.")

        for t in reversed(range(0, num_steps)):
            t_tensor = torch.full(
                (shape[0],), t, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = self.unet(x, t_tensor, aim, cond_features)

            alpha = 1 - self.beta[t]
            alpha_bar = self.alpha_bar[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) /
                                              torch.sqrt(1 - alpha_bar)) * pred_noise)
            x += torch.sqrt(self.beta[t]) * noise

        return x.clamp(-1, 1)
