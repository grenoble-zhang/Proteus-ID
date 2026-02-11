<div align ="center">
<h1> Proteus-ID </h1>
<h3> Proteus-ID: ID-Consistent and Motion-Coherent Video Customization </h3>
<div align="center">
</div>

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://grenoble-zhang.github.io/Proteus-ID/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2506.23729-b31b1b.svg)](https://arxiv.org/abs/2506.23729)&nbsp;
</div>

Authors: [Guiyu Zhang](https://grenoble-zhang.github.io/)<sup>1</sup>, [Chen Shi](https://scholar.google.com.hk/citations?user=o-K_AoYAAAAJ&hl=en)<sup>1</sup>, Zijian Jiang<sup>1</sup>, Xunzhi Xiang<sup>2</sup>, Jingjing Qian<sup>1</sup>, [Shaoshuai Shi](https://shishaoshuai.com/)<sup>3</sup>, [Li Jiang†](https://llijiang.github.io/)<sup>1</sup>

<sup>1</sup> The Chinese University of Hong Kong, Shenzhen&emsp;<sup>2</sup> Nanjing University&emsp;
<sup>3</sup> Voyager Research, Didi Chuxing


<img src="assets\representative_image.jpg" width="100%"/>

## TODO

- [x] Release arXiv technique report
- [x] Release full codes
- [ ] Release dataset (coming soon)

## 🛠️ Requirements and Installation
### Environment

```bash
# 0. Clone the repo
git clone --depth=1 https://github.com/grenoble-zhang/Proteus-ID.git

cd /nfs/dataset-ofs-voyager-research/guiyuzhang/Opensource/code/Proteus-ID-main

# 1. Create conda environment
conda create -n proteusid python=3.11.0
conda activate proteusid

# 3. Install PyTorch and other dependencies
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# 4. Install pip dependencies
pip install -r requirements.txt
```

### Download Model

```bash
cd util
python download_weights.py
python down_raft.py
```

Once ready, the weights will be organized in this format:
```
🔦 ckpts/
├── 📂 face_encoder/
├── 📂 scheduler/
├── 📂 text_encoder/
├── 📂 tokenizer/
├── 📂 transformer/
├── 📂 vae/
├── 📄 configuration.json
├── 📄 model_index.json
```

## 🏋️ Training

```bash
# For single rank
bash train_single_rank.sh
# For multi rank
bash train_multi_rank.sh
```

## 🏄️ Inference

```bash
python inference.py --img_file_path assets/example_images/1.png --json_file_path assets/example_images/1.json
```


## BibTeX
If you find our work useful in your research, please consider citing our paper:
```bibtex
@inproceedings{zhang2025proteus,
  title={Proteus-id: Id-consistent and motion-coherent video customization},
  author={Zhang, Guiyu and Shi, Chen and Jiang, Zijian and Xiang, Xunzhi and Qian, Jingjing and Shi, Shaoshuai and Jiang, Li},
  booktitle={Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
  pages={1--11},
  year={2025}
}
```

## Acknowledgement

Thansk for these excellent opensource works and models: [CogVideoX](https://github.com/THUDM/CogVideo); [ConsisID](https://github.com/PKU-YuanGroup/ConsisID); [diffusers](https://github.com/huggingface/diffusers).
