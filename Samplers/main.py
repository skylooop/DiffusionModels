import pyrallis
from dataclasses import dataclass, field
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
from ddpm import DDPM
import numpy as np

def sample_image(pipe, sampler, prompt, init_latents, batch_size, guidance):
    with torch.inference_mode():
        latents = init_latents
        cond_input = pipe.tokenizer([prompt], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = pipe.text_encoder(cond_input.input_ids.to(pipe.device))[0]
        
        uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        for t in tqdm(sampler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(latent_model_input, t, text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
            latents = sampler(noise_pred, latents, t)
            
        with torch.autocast('cuda'):
            images = pipe.vae.decode(latents * 1 / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = np.uint8(255 * images)

        return images


@dataclass
class global_options:
    sampler: str = field()
    prompt: str = field()

    def __post_init__(self):
        print(f"Prompt: {self.prompt}, Sampler: {self.sampler}")

@pyrallis.wrap()
def main(options: global_options):
    print("LOADING TOKEN and SD model")
    load_dotenv()

    token = os.getenv("HUG_TOKEN")
    model_id = "runwayml/stable-diffusion-v1-5"
    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=token, torch_dtype=torch.float16, revision="fp16").cuda()
    
    batch_size = 1
    init_latents = torch.randn(batch_size, 4, 64, 64).cuda()
    
    if global_options.sampler == "DDPM":
        sampler = DDPM()
    image = sample_image(model, sampler, options.prompt, init_latents)
    
    
if __name__ == "__main__":
    main()    



