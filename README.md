<div align="center">
<h2>Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data</h2>

[**Zehan Wang**]()<sup>1*</sup> · [**Ziang Zhang**]()<sup>1*</sup> · [**Tianyu Pang**]()<sup>2</sup> · [**Du Chao**]()<sup>2</sup> · [**Hengshuang Zhao**]()<sup>3</sup> · [**Zhou Zhao**]()<sup>1</sup>

<sup>1</sup>Zhejiang University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>SEA AI Lab&emsp;&emsp;&emsp;&emsp;<sup>3</sup>HKU

*Equal Contribution


<a href=""><img src='https://img.shields.io/badge/arXiv-Orient Anything-red' alt='Paper PDF'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-Orient Anything-green' alt='Project Page'></a>
<a href='g'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a>
</div>

****Orient Anything**, a robust image-based object orientation estimation model. By training on 2M rendered labeled images, it achieves strong zero-shot generalization ability for images in the wild.

![teaser](assets/demo.png)

## News

* **2024-12-24:** Paper, project page, code, models, and demo ([HuggingFace](https://huggingface.co/spaces/LiheYoung/Depth-Anything)) are released.



## Pre-trained models

We provide **three models** of varying scales for robust object orientation estimation in images:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Orient-Anything-Small | xxM | [Download]() |
| Orient-Anything-Base | xxM | [Download]() |
| Orient-Anything-Large | xxxM | [Download]() |

## Usage

### Prepraration

```bash
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

### Best Practice
为了避免歧义，我们的模型只支持包含单个物体的图像的输入。对于通常包含多个物体的图像，结合DINO-grounding来isolate出各个物体，分别预测朝向是一个好的选择。

```python
[ToDo]怎么结合DINO-grounding分离物体
```

### Test-Time Augmentation
为了进一步增强模型的鲁棒性，我们进一步提出test-time ensemble策略。输入的图像会被随机裁剪，变成不同的variant，对不同变体预测出来的朝向被vote作为最终的预测结果。
```python
[ToDo]怎么做test-time ensemble
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{orientanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```