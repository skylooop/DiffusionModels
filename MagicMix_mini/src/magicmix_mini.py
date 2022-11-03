import torch
import pyrallis
import numpy as np
from PIL import Image
import os
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel, 
    StableDiffusionImg2ImgPipeline,
)
from tqdm import tqdm
from torchvision.transforms import ToTensor

def MagicMix(get_processed_latents: list[torch.Tensor],
             scheduler,
             vae,
             unet,
             text_embedding: torch.Tensor,
             uncond_emb: torch.Tensor,
             nu: float,
             K_min: int,
             K_max: int,
             scale: float,
             reverse_steps: int):
    
    init_timestep = min(K_min, reverse_steps)
    t_start = max(reverse_steps - init_timestep, 0)
    
    first = get_processed_latents[0] #not so many noise
    prompts_embeddings = torch.cat([uncond_emb, text_embedding])
    
    timesteps = scheduler.timesteps[t_start:].to("cuda")
    for i, t in enumerate(tqdm(timesteps)):
        latent_model_input = torch.cat([first] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompts_embeddings)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        first = scheduler.step(noise_pred, t, first).prev_sample
        
        if i < len(get_processed_latents):
            first = first * nu + (1 - nu) * get_processed_latents[i]
        
    del get_processed_latents
    return first        
        
        
def prepare_models():
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        use_auth_token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16,
    ).to(torch.device("cuda"))

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch.device("cuda"))

    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_auth_token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16,
    ).to(torch.device("cuda"))

    return vae, unet, text_encoder, tokenizer

def pil_to_latent(input_im, vae):
    latent = vae.encode(ToTensor()(input_im).unsqueeze(0).to(torch.device("cuda")) * 2 - 1)
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(vae, latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def make_scheduler_latents(
    scheduler,
    sampled_latents,
    noise,
    K_min,
    K_max,
    reverse_steps
) -> list[torch.Tensor]:
    
    start_of_embed = K_min
    processed_latents = []
    
    for i in range(K_max - K_min):
        step = start_of_embed + i
        timesteps = scheduler.timesteps[-step] #get from lowest to highest
        timesteps = torch.tensor([timesteps] * noise.shape[0]).to(noise.device)
        
        diffusion_init_latent = sampled_latents.clone()
        diffusion_init_latent = scheduler.add_noise(
            diffusion_init_latent, noise, timesteps 
        )
        
        processed_latents.append(diffusion_init_latent)
    
    return list(reversed(processed_latents))

@torch.no_grad()
@torch.autocast("cuda")
def semantic_mixture(params):
    
    image = Image.open(params.input_image).convert("RGB")
    
    image = image.resize((576, 576))
    vae, unet, text_encoder, tokenizer = prepare_models()
    sampled_from_latent = pil_to_latent(image, vae) #sample from prior distribution
    
    height, width = torch.tensor(np.asarray((image))).shape[:-1]
    noise = torch.randn(
        (len(params.prompts), unet.in_channels, height // 8, width // 8)
    ).to(torch.device("cuda"))
    
    #MagicMix
    get_processed_latents = make_scheduler_latents(
        params.scheduler, sampled_from_latent, noise,
        params.K_min, params.K_max, params.reverse_steps
    )
    process_semantics = tokenizer(params.prompts,
                                  padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True,
                                  return_tensors="pt")
    
    text_embeddings = text_encoder(process_semantics.input_ids.to("cuda"))[0]
    max_length = process_semantics.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * len(params.prompts), padding="max_length", max_length=max_length, return_tensors="pt"
    )
    
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0] 
    
    #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    mix = MagicMix(
        get_processed_latents,
        params.scheduler,
        vae,
        unet,
        text_embeddings,
        uncond_embeddings,
        params.nu,
        params.K_min,
        params.K_max,
        params.guidance,
        params.reverse_steps
    )
    
    return latents_to_pil(vae, mix)
    
    