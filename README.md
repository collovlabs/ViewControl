# Integrating View Conditions for Image Synthesis
This is the official implementation of the paper "Integrating View Conditions for Image Synthesis", which is accepted by IJCAI 2024. 🎉 

[[Paper]](https://www.ijcai.org/proceedings/2024/840)

## Introduction

This paper presents **ViewControl** that enhances existing models with awareness of viewpoint information, thereby
enabling improved control over text-to-image diffusion models, such as Stable Diffusion. This advancement leads to a
more controllable approach for image editing tasks. Our proposed pipeline effectively addresses crucial aspects of image synthesis, including *consistency*, *controllability*, and *harmony*. Through both quantitative and qualitative comparisons with recently published
open-source state-of-the-art methods, we have showcased the
favorable performance of our approach across various dimensions.


## Pipeline

The pipeline of ViewControl consists of three steps: LLM Planer, Pose Estimation and Synthesis, and Image Synthesis. The LLM Planer is responsible for understanding the users' inputs and bridging the gap between the users' inputs and the following steps. The Pose Estimation and Synthesis module is responsible for estimating the pose of the object in the input image and synthesizing the image of the object at the target pose. The Image Synthesis module is responsible for synthesizing the final image by combining the synthesized image of the object with the background of the input image. The pipeline of ViewControl is shown in the following figure:

<p align="center">
  <img src="./imgs/demo/pipeline.jpg" width="800" />

## Installation
First, clone the repository locally:
```bash
git clone https://github.com/huggingface/diffusers.git 
git clone https://github.com/luca-medeiros/lang-segment-anything.git 
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
Then, create a conda environment and install the required packages:
```bash
conda create -n view_cond python=3.10
conda activate view_cond

cd diffusers
pip install -e .

cd ../lang-segment-anything
pip install torch torchvision
pip install -e .

cd ../GroundingDINO
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

cd ..

pip install -r requirements.txt
cd pose_synthesis
python download_ckpt.py

pip install --upgrade torchaudio
cd ..
```

## Training
If you want to train your own pose estimator, you can use the following command:
```bash
python train_pose_estimator.py --dataset_path <path_to_dataset> --output_dir <path_to_output_dir>
```
You may need to adjust the hyperparameters (learning rate, batch size, etc.) in the script to get the best performance.

Your dataset should be organized as follows:
```
dataset
├── class_1
│   ├── obj_1
│   │   ├── x1_y1.png
│   │   ├── x2_y2.png
│   │   ├── ...
│   │   └── xN_yN.png
│   ├── obj_2
│   │   ├── x1_y1.png
│   │   ├── x2_y2.png
│   │   ├── ...
│   │   └── xN_yN.png
│   ├── ...
│   └── obj_N
│       ├── ...
|── class_N
│   ├── ...
```
where `x1_y1.png` is the image of `obj_1` at pose `(x1, y1)`, and `class_1` is the class name of `obj_1`. The dataset can be synthetic or real. If you want to synthesize your own dataset, first prepare a set of images of the object for the same class with the same pose, then use the pose_synthesis module to synthesize the images of the object at different poses. 


## Inference

### Pose Estimation
To estimate the pose of a given image, you can use the following command:
```bash
python pose_estimation.py --image_path <path_to_image> --output_dir <path_to_output_dir> --model_path <path_to_model>
```
### Pose Synthesis
To synthesize an image of one object from a given pose, you can use the following command:
```bash
cd pose_synthesis
python pose_synthesis_batch.py --input_dir input_dir --output_dir output_dir --x x_value --y y_value
cd ..
```
for example:
```bash
cd pose_synthesis
python pose_synthesis_batch.py --input_dir input_dir --output_dir output_dir --x 0 --y 0
cd ..
```
To synthesis a set of images of one object from a given set of poses, you can use the following command:
```bash
cd pose_synthesis
python pose_synthesis_batch.py --input_dir input_dir --output_dir output_dir --x x_values --y y_values
cd ..
```
for example:
```bash
cd pose_synthesis
python pose_synthesis_batch.py --input_dir input_dir --output_dir output_dir --x 0,10 --y 0,-10
cd ..
```
### Image Synthesis
To synthesize an image, you can use the following command:
```bash
python image_synthesis.py --path_src_img <path_to_src_img> --path_ref_img <path_to_ref_img> --text_prompt <text_prompt> --save_path <save_path> --mask_obj_name <mask_obj_name> --ref_obj_name <ref_obj_name>
```
If you need faster inference, you can set the dreambooth option to False or pre train it or change it to another lightweight personalization method like LoRA. 

### Other Utils
If you need to obtain a more accurate caption or class name from an image, you can use the following command:
```bash
python obj_name_synthesis.py --path_src_img <path_to_src_img> --save_path <save_path>
```

If you need to remove the background of an image, you can use the following command:
```bash
python utils.py --input_path <input_path> --prompt <prompt> --output_path <output_path>
```

## Examples
Here are some examples of the results of ViewControl:
<p align="center">
  <img src="./imgs/demo/Intro.jpg" width="800" />
</p>


## Citation 
If you find this work useful, please cite our paper:
```
@inproceedings{ijcai2024p840,
  title     = {Integrating View Conditions for Image Synthesis},
  author    = {Bai, Jinbin and Dong, Zhen and Feng, Aosong and Zhang, Xiao and Ye, Tian and Zhou, Kaicheng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {7591--7599},
  year      = {2024},
  month     = {8},
  note      = {AI, Arts & Creativity},
  doi       = {10.24963/ijcai.2024/840},
  url       = {https://doi.org/10.24963/ijcai.2024/840},
}
```
