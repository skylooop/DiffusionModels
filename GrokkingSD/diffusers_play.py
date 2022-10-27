'''
First login with `huggingface-cli login`
'''

import torch
import torchvision
from tqdm.auto import tqdm
from torch.cuda.amp.autocast_mode import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from diffusers_utils import UNet2DConditionModel, LMSDiscreteScheduler, Models
import pyrallis
from dataclasses import dataclass, field

device = torch.device("cuda")
to_tensor_tfm = torchvision.ToTensor()

def pil_to_latent(input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = Models.vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to(device)*2-1) # Note scaling
  return 0.18215 * latent.mode() # or .mean or .sample

def latents_to_pil(latents):
  # bath of latents -> list of images
  latents = (1 / 0.18215) * latents
  
  with torch.no_grad():
    image = Models.vae.decode(latents)
    
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  pil_images[0].save(fp="/home/m_bobrin/Text2imgScratch/text2img/examples/ex.jpg")


def Display(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(fp="/home/m_bobrin/Text2imgScratch/text2img/examples/ex.jpg")

@dataclass
class Params:
    prompt: str = "A watercolor painting of a macaw"
    height: int = 512                        
    width:  int = 512                        
    num_inference_steps: int = 30            # Num of backward steps
    guidance_scale: int = 7.5                # Scale for classifier-free guidance
    #generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
    batch_size: int = 1
    

def main(args: Params):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    vae = Models.vae.to(device)
    text_encoder = Models.text_encoder.to(device)
    unet = Models.unet.to(device)
    unet = unet.to(device) # 4 channels, 3 rgb, 1 for text embedding
    
    # Classifier free-guidance
    text_input = Models.tokenizer(args.prompt, padding="max_length",
                           max_length=Models.tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input['input_ids'].to(device))[0] #cond_embed
    max_length = text_input['input_ids'].shape[-1]
    
    #Unconditional embeddings
    uncond_input = Models.tokenizer(
        [""] * args.batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    ) # size (1, max_length)
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input['input_ids'].to(device))[0] 
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    scheduler.set_timesteps(args.num_inference_steps)
    
    latents = torch.randn(
        (args.batch_size, unet.in_channels, args.height // 8, args.width // 8))
    #(1, 4, 64, 64)
    latents = latents.to(device)
    latents = latents * scheduler.sigmas[0]
    
    with autocast():
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2) #shape (2, 4, 64, 64)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
                
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents)
    
    # Display
    Display(image.sample)
    
    

if __name__ == "__main__":
    args = pyrallis.parse(config_class=Params)
    main(args)