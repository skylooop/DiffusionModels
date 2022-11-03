from email.policy import default
import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import field, dataclass
import pyrallis
from typing import List
from magicmix_mini import semantic_mixture
from diffusers import LMSDiscreteScheduler

@dataclass
class MagicMixCFG:
    input_image: Path = None
    exper_dir: Path = field(default=Path("/home/m_bobrin/DiffusionModels/MagicMix_mini/addons"))
    reverse_steps: int = field(default=50)
    #Number of steps (in paper K_min = k_min * reverse_steps)
    k_min: float = field(default=0.3)
    k_max: float = field(default=0.6)
    #Bigger -> more like source image
    nu: float = field(default=0.7)
    guidance: float = field(default=8)
    prompts = ["corgi", "coffee machine"]
    
    def __post_init__(self):
        self.K_min = self.reverse_steps * self.k_min
        self.K_max = self.reverse_steps * self.k_max
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )
        self.scheduler.set_timesteps(self.reverse_steps)

@pyrallis.wrap()
def main(MM_cfg: MagicMixCFG):
    load_dotenv(verbose=True)
    
    mixed_images = semantic_mixture(MM_cfg)




if __name__ == "__main__":
    main()