import pyrallis
from dataclasses import dataclass, field
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import os
from ddpm import DDPM

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
    
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=token).cuda()
    if global_options.sampler == "DDPM":
        sampler = DDPM()

if __name__ == "__main__":
    main()    



