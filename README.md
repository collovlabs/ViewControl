# Intergrating View Conditions for Image Synthesis
This is the official implementation of the paper "Intergrating View Conditions for Image Synthesis". 
[[Paper]](https://arxiv.org/pdf/2310.16002v3)

## Pipeline
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
Our pipeline consists of three steps: pose estimation, pose synthesis and image synthesis.
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

### Others
If you need to obtain a more accurate caption or class name from an image, you can use the following command:
```bash
python obj_name_synthesis.py --path_src_img <path_to_src_img> --save_path <save_path>
```

If you need to remove the background of an image, you can use the following command:
```bash
python utils.py --input_path <input_path> --prompt <prompt> --output_path <output_path>
```

## Examples
Here are some examples of our method:
<p align="center">
  <img src="./imgs/demo/Intro.jpg" width="800" />
</p>


## Citation 
If you find this work useful, please cite our paper:
```
@article{bai2023integrating,
  title={Integrating view conditions for image synthesis},
  author={Bai, Jinbin and Dong, Zhen and Feng, Aosong and Zhang, Xiao and Ye, Tian and Zhou, Kaicheng},
  journal={arXiv preprint arXiv:2310.16002},
  year={2023}
}
```