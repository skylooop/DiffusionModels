import torch
import torchvision
from tqdm.auto import tqdm
from torch.cuda.amp.autocast_mode import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from diffusers_utils import UNet2DConditionModel, LMSDiscreteScheduler, Models

device = torch.device("cuda")

def main():
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    vae = Models.vae.to(device)
    text_encoder = Models.text_encoder.to(device)
    unet = unet.to(device)
    

if __name__ == "__main__":
    main()