import torch
import os
from torchvision.models.optical_flow import Raft_Large_Weights

def download_raft_weights():
    weights = Raft_Large_Weights.DEFAULT
    url = weights.url
    
    # Get cache directory
    cache_dir = os.path.expanduser('../raft_weight/')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get filename from URL
    filename = url.split('/')[-1]
    filepath = os.path.join(cache_dir, filename)
    
    print(f"Downloading: {url}")
    print(f"Saving to: {filepath}")
    
    # Download file
    torch.hub.download_url_to_file(url, filepath, progress=True)
    print("Download completed!")

download_raft_weights()