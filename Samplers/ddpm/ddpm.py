import numpy as np
import torch
from typing import Any

def get_beta_schedule(num_diffusion_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
    return betas

class DDPM:
    
    def __init__(self, num_train_steps: int = 1000, num_inference_steps: int = 500, reverse_sample: bool = True) -> None:
        betas = get_beta_schedule(num_train_steps + 1)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        
        self.divide = len(alphas) // num_inference_steps
        self.timesteps = torch.arange(num_train_steps + 1)        
        
        betas = torch.clamp(betas, -torch.inf, 0.99)
        alpha_bar = torch.cos((self.timesteps/(num_train_steps+1) + 0.008)/(1 + 0.008) * torch.pi/2) #Transformer positional encoding
        
        self.beta = {t.item(): beta for t, beta in zip(self.timesteps, betas)}
        self.alpha = {t.item(): alpha for t, alpha in zip(self.timesteps, betas)}
        self.alpha_bar = {t.item(): alpha_bar for t, alpha_bar in zip(self.timesteps, alpha_bar)}
        
        self.timesteps = self.timesteps[::self.divide]
        
        if reverse_sample: 
            self.timesteps = reversed(self.timesteps)[:-1]

        self.reverse_sample = reverse_sample

    def __call__(self, eps_theta, x, t) -> Any:
        t = t.item()
        tprev = t - self.stride if self.reverse_sample else t + self.stride

        beta, alpha = self.beta[t], self.alpha[t]
        alpha_bar, alpha_bar_prev = self.alpha_bar[t], self.alpha_bar[tprev]

        sigma = np.sqrt(beta)
        beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta  # eqn (7)
        
        # algorithm 2
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x_prev = 1 / np.sqrt(alpha) * (x - (1 - alpha) / (1 - alpha_bar) * eps_theta) + sigma * z

        return x_prev
    
    def __repr__(self) -> str:
        return f"{self.betas}"
    
print(DDPM())