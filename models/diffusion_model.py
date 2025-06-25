import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

class AdvCEODiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=config['patch_size'],
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            )
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config['num_diffusion_steps'],
            beta_schedule="linear"
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timesteps).sample

    def add_noise(self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.noise_scheduler.add_noise(clean_images, noise, timesteps)