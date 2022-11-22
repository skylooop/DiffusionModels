import numpy as np
import torch


def get_beta_schedule(num_diffusion_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
    return betas

class DDPM:
    
    def __init__(self, num_train_steps: int = 1000, num_inference_steps: int = 500, reverse_sample: bool = True) -> None:
        betas = get_beta_schedule(num_train_steps + 1)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(betas[alphas],0)
        
        
        
    
    pass