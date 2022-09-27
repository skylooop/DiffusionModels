import requests
import typing as tp
import os


def downloader(*args, **kwargs) -> None:
    #####  Dataset Link - https://data.vision.ee.ethz.ch/cvl/DIV2K/  #####
    URLS = ["http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"]
    BASE_DIR = "/home/m_bobrin/GANimplementations/data"
    print("Downloading DIV2K Dataset ± 5 GB for training")
    for file in URLS:
        name_file = file.split("/")[-1]
        print(f"Downloading {name_file}")  
        curr_path = os.path.join(BASE_DIR, name_file)
        os.makedirs(curr_path, exist_ok=True)     
        req = requests.get(file)
    print("Downloading Completed")
    
#if __name__ == "__main__":
    #downloader()
        

