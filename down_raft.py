import torch
import os
from torchvision.models.optical_flow import Raft_Large_Weights

def download_raft_weights():
    weights = Raft_Large_Weights.DEFAULT
    url = weights.url
    
    # 获取缓存目录
    cache_dir = os.path.expanduser('/nfs/voyager-research-hdd/users/shichen/project/guiyu/ConsisID/proteus/v2_motion/raft_weight/')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 从URL获取文件名
    filename = url.split('/')[-1]
    filepath = os.path.join(cache_dir, filename)
    
    print(f"正在下载: {url}")
    print(f"保存到: {filepath}")
    
    # 下载文件
    torch.hub.download_url_to_file(url, filepath, progress=True)
    print("下载完成！")

download_raft_weights()