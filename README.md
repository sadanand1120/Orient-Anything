<div align="center">
<h2>Orient Anything: Learning Robust Object Orientation Estimation from Rendering 3D Models</h2>

[**Zehan Wang**](https://scholar.google.com/citations?user=euXK0lkAAAAJ&hl=zh-CN)<sup>1*</sup> · [**Ziang Zhang**](https://scholar.google.com/citations?hl=zh-CN&user=DptGMnYAAAAJ)<sup>1*</sup> · [**Tianyu Pang**](https://scholar.google.com/citations?hl=zh-CN&user=wYDbtFsAAAAJ)<sup>2</sup> · [**Du Chao**](https://scholar.google.com/citations?hl=zh-CN&user=QOp7xW0AAAAJ)<sup>2</sup> · [**Hengshuang Zhao**](https://scholar.google.com/citations?user=4uE10I0AAAAJ&hl&oi=ao)<sup>3</sup> · [**Zhou Zhao**](https://scholar.google.com/citations?user=IIoFY90AAAAJ&hl&oi=ao)<sup>1</sup>

<sup>1</sup>Zhejiang University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>SEA AI Lab&emsp;&emsp;&emsp;&emsp;<sup>3</sup>HKU

*Equal Contribution


<a href=""><img src='https://img.shields.io/badge/arXiv-Orient Anything-red' alt='Paper PDF'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-Orient Anything-green' alt='Project Page'></a>
<a href='g'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a>
</div>

**Orient Anything**, a robust image-based object orientation estimation model. By training on 2M rendered labeled images, it achieves strong zero-shot generalization ability for images in the wild.

![teaser](assets/demo.png)

## News

* **2024-12-24:** Paper, project page, code, models, and demo ([HuggingFace](https://huggingface.co/spaces/Viglong/Orient-Anything)) are released.



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

```

### Best Practice
To avoid ambiguity, our model only supports input of images containing a single object. For images that contain multiple objects, it is a good choice to combine DINO-grounding and SAM to isolate each object and predict the orientation separately.

```python
[ToDo]
```

### Test-Time Augmentation
In order to further enhance the robustness of the model, we further propose a test-time ensemble strategy. The input image will be randomly cropped into different variants, and the orientations predicted for different variants will be voted as the final prediction results.
```python
[ToDo]
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{orient_anything,
  title={Orient Anything: Learning Robust Object Orientation Estimation from Rendering 3D Models},
  author={Wang, Zehan and Zhang, Ziang and Pang, Tianyu and Du, Chao and Zhao, Hengshuang and Zhao, Zhou},
  journal={arXiv:2412.xxxxx},
  year={2024}
}
```