import torch
import numpy
import einops
import dataset
from dataclasses import dataclass, field
import pyrallis

@dataclass
class TrainConfig:
    # Whether to download dataset for training
    train: field(default=True)

    #def __post_init__():
        # if no default - uncomment
           
def main():
    cfg = pyrallis.parse(config_class=TrainConfig)
    if cfg.train:
        dataset.downloader()
        
        
if __name__ == "__main__":
    main()
    