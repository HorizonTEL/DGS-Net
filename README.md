<div align="center">
 <br>
<h1>DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection
</h1>
 
[Jiazhen Yan](https://scholar.google.com/citations?user=QkURh8EAAAAJ&hl=zh-CN)<sup>1</sup>, [Ziqiang Li](https://scholar.google.com/citations?user=mj5a8WgAAAAJ&hl=zh-CN)<sup>1</sup>,  [Fan Wang](https://scholar.google.com/citations?user=zT1Ad0gAAAAJ&hl=zh-CN)<sup>2</sup>, [Boyu Wang](https://scholar.google.com/citations?user=YvqxcCQAAAAJ&hl=zh-CN)<sup>1</sup>, [Ziwen He](https://scholar.google.com/citations?user=PjkDK9cAAAAJ&hl=zh-CN)<sup>1</sup>, [Zhangjie Fu](https://scholar.google.com/citations?user=fO9NmagAAAAJ&hl=zh-CN)<sup>1‡</sup>


<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>Nanjing University of Information Science and Technology
<sup>2</sup>University of Macau
<p align="center">
  <a href='https://github.com/HorizonTEL/DGS-Net'>
    <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://arxiv.org/abs/2511.13108'>
    <img src='https://img.shields.io/badge/Arxiv-2511.13108-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2511.13108'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
</p>
</div>
<img width="7109" height="2343" alt="backbone" src="https://github.com/user-attachments/assets/862bb444-2cbe-4a66-8e2b-cf072a6aee38" />

## 📰 News
* [2026-05-01]🎉🎉🎉 DGS-Net is accepted by ICML 2026 Spotlight
## 🔥 Todo List
- [x] **Inference code**
- [x] **Pretrained models**
- [x] **Training code**
## 🚀 Quick Start
### 1. Installation
```
conda create -n DGS-Net -y python=3.10

conda activate DGS-Net
pip3 install torch torchvision
pip install -r requirements.txt 
```
### 2.Getting datasets
| Datasets          |    Paper                                                                                                             |    Url    |
|:------:           |:---------:                                                                                                             |:---------:|
| UniversalFakeDetect| Towards Universal Fake Image Detectors that Generalize Across Generative Models (CVPR 2023)                            | [Google Drive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-) |
| AIGCDetectBench   | PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection                                         | [ModelScope](https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/tree/master/AIGCDetectionBenchMark) |
| AIGIBench         | Is Artificial Intelligence Generated Image Detection a Solved Problem? (NeurIPS 2025)                                  | [Huggingface](https://huggingface.co/datasets/HorizonTEL/AIGIBench)/[Baidu Netdisk](https://pan.baidu.com/s/1XTwfXlfqkGxAwYLxXuZbfA?pwd=sm6v) |
### 3.Download Weights
Place them under ```checkpoints/```:
[Google Drive](https://drive.google.com/drive/folders/1_MznDWLcZDNCeWj6AeP8MrJBLoMomxe7?usp=drive_link)
### 4.Inference
Of course, you need to change [DetectionTests] in test.py when testing.

We also present our inference results in log_test.log.
```[Note]: Following re-inference, we observed an approximately 0.1 discrepancy relative to the original result. The file log_test.log contains the results obtained after re-inference.```
```
python test.py --model_path ./checkpoints/model_epoch_step2.pth
```
## ⏳ Training
The training set uses ProGAN & SDv1.4 from AIGIBench (Is Artificial Intelligence Generated Image Detection a Solved Problem?, NeurIPS 2025) [Huggingface](https://huggingface.co/datasets/HorizonTEL/AIGIBench)/[Baidu Netdisk](https://pan.baidu.com/s/1XTwfXlfqkGxAwYLxXuZbfA?pwd=sm6v).
```
stage 1: python train.py --name  5class-car-cat-chair-horse-sdv1.4 --dataroot /home/HDD/yjz/dataset/AIGIBench/train  --classes  car,cat,chair,horse,sdv1.4 --train_stage 0 --niter 3
stage 2: python train.py --name  5class-car-cat-chair-horse-sdv1.4 --dataroot /home/HDD/yjz/dataset/AIGIBench/train  --classes  car,cat,chair,horse,sdv1.4 --train_stage 1 --niter 1
```
During the stage 2 of training, it is necessary to change the model_path of stage 1 in ```models/clip_models```: ```self.clip_model_frozen.load_state_dict(torch.load("/path/checkpoints/model_epoch_stage1.pth"), strict=True)```
## 📚 Citation 
```
@article{yan2025dgs,
  title={DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection},
  author={Yan, Jiazhen and Li, Ziqiang and Wang, Fan and Wang, Boyu and He, Ziwen and Fu, Zhangjie},
  journal={arXiv preprint arXiv:2511.13108},
  year={2025}
}
```

## 📬 Contact
If you have any question about this project, please feel free to contact 247918horizon@gmail.com
