'''
Prompt-to-Prompt Image Editing with Cross Attention Control 
https://arxiv.org/pdf/2208.01626.pdf

Method:
Run standard diffusion using text \mathcal{P} and get image \mathcal{I}
'''
import os
import wandb
import numpy as np
import torch
import torchvision
from transformers import CLIPModel, CLIPTokenizer, CLIPTextModel
from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        LMSDiscreteScheduler
)
import pyrallis
from dataclasses import field, dataclass

@dataclass
class Wandb_CFG:
    wandb_project: str = field(default="CrossAttentionControl")
    model_precision: str = field(default="fp16")
    model_path_clip: str = field(default="openai/clip-vit-large-patch14")
    model_path_diffusion: str = field(default="CompVis/stable-diffusion-v1-4")
    device: torch.device = torch.device("cuda")

def load_models(config: wandb.config) -> None:
    print("Loading Models")
    
    clip_tokenizer = CLIPTokenizer.from_pretrained(config.model_path_clip)
    clip_model = CLIPModel.from_pretrained(
        config.model_path_clip,
        torch_dtype=torch.float16
    )
    clip = clip_model.text_model.to(config.device)
    
    unet = UNet2DConditionModel.from_pretrained(
        config.model_path_diffusion,
        subfolder="unet",
        revision=config.model_precision,
        torch_dtype=torch.float16
    ).to(config.device)
    vae = AutoencoderKL.from_pretrained(
        config.model_path_diffusion,
        revision=config.model_precision,
        torch_dtype=torch.float16
    ).to(config.device)

def init_wandb(args: Wandb_CFG) -> None:
    '''
                Init Logger
    '''
    wandb.wandb_init(args.wandb_project)
    config = wandb.config
    
    '''
                Init Models
    '''
    load_models(config)
    
    

if __name__ == "__main__":
    args = pyrallis.parse(config_class=Wandb_CFG)
    init_wandb(args)

