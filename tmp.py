from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline,UNet2DConditionModel,DDIMScheduler,StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import cv2
import os
import numpy as np
import json
import time
import traceback
import gradio as gr
import logging
import subprocess
# from PIL import Image
from PIL import Image as _Image
import PIL
import PIL.ImageOps
# Image.init()
from lang_sam import LangSAM # cd ./lang-segment-anything; pip3 install -e . #cd ./GroundingDINO; pip install -e . # https://github.com/IDEA-Research/GroundingDINO/issues/8 for NameError: name '_C' is not defined
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
import argparse


# config

port=8046
# os.environ['CUDA_VISIBLE_DEVICES']
device = "cuda:0"
torch.cuda.empty_cache()
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'


controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16),
]
unet = UNet2DConditionModel.from_pretrained('/opt/disk-sdc/jinbin/dreambooth_models/sofa_20_a/unet', torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, unet=unet, torch_dtype=torch.float16, safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()


def bbox(image, text):

    model = LangSAM(sam_type="vit_h")
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model.device:", model.device)

    def draw_image(image, masks, boxes, labels, alpha=0.4):
        image = torch.from_numpy(image).permute(2, 0, 1)
        if len(boxes) > 0:
            image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
        if len(masks) > 0:
            image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
        return image.numpy().transpose(1, 2, 0)

    def predict(image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        if isinstance(image_path, str):
            image_pil = _Image.open(image_path).convert("RGB")
        else:
            # bug here, need to be improved
            image_pil = image_path
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        image_array = np.asarray(image_pil)
        image = draw_image(image_array, masks, boxes, labels)
        image = _Image.fromarray(np.uint8(image)).convert("RGB")

        mask_image1 = draw_image(np.zeros_like(image_array), masks, [], [], alpha=1.0)
        mask_image1 = _Image.fromarray(np.uint8(mask_image1)).convert("RGB")

        boxes_mask = torch.zeros_like(masks)
        new_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1) - 1, int(y1) - 1, int(x2) + 1, int(y2) + 1
            new_boxes.append([x1, y1, x2, y2])
            boxes_mask[:, y1:y2, x1:x2] = 1

        mask_image2 = draw_image(np.zeros_like(image_array), boxes_mask, [], [], alpha=1.0)
        mask_image2 = _Image.fromarray(np.uint8(mask_image2)).convert("RGB")
        boxes = torch.tensor(new_boxes)
        shape_mask_image = draw_image(np.zeros_like(image_array), masks, [], [], alpha=1.0)
        shape_mask_image = _Image.fromarray(np.uint8(shape_mask_image)).convert("RGB")
        return mask_image2, boxes, image, shape_mask_image

    mask_image, boxes, image, shape_mask_image = predict(image, text)
    return mask_image, boxes, shape_mask_image


def predict_v1(src_img=None, ref_img=None, mask_obj_name="", ref_obj_name="", mask_type="auto mask", controlnet_scale_1=0.8, controlnet_scale_2=0.3, src_msk_img=None):
    start = time.time()
    torch.cuda.empty_cache()
    text_prompt = ref_obj_name
    print("start test process")
    print("type of src_img", type(src_img))

    draw_mask = None
    if type(src_img) is dict:
        draw_mask = src_img['mask']
        src_img = src_img['image']


    #logging.info(mask_obj_name)
    #logging.info(ref_obj_name)
    #logging.info(text_prompt)

    #image = _Image.fromarray(cv2.resize(np.array(src_img), (512, 512)))
    #canny_image = _Image.fromarray(cv2.resize(np.array(ref_img), (512, 512)))
    ##mask_image, boxes, shape_mask_image = bbox(image, mask_obj_name)
    ##mask_image_2, boxes_2, shape_mask_image_2 = bbox(canny_image, ref_obj_name)
    #output_image = image
    #end = time.time()
    ##logging.info("test process")
    #print("test process")
    #return [output_image]


    try:
        image = _Image.fromarray(cv2.resize(np.array(src_img), (512, 512)))
        canny_image = _Image.fromarray(cv2.resize(np.array(ref_img), (512, 512)))
        
        mask_image, boxes, shape_mask_image = bbox(image, mask_obj_name)
        mask_image_2, boxes_2, shape_mask_image_2 = bbox(canny_image, ref_obj_name)

        
        # get the bbox number of the mask, when then is only one box
        x1, y1, x2, y2 = boxes[0]
        x1_2, y1_2, x2_2, y2_2 = boxes_2[0]

        # adjust the canny image, so that the bbox of the mask is the same as the bbox of the canny image
        canny_image = np.array(canny_image)
        color_image = canny_image
        
        shape_mask_image = np.array(shape_mask_image)
        shape_mask_image_2 = np.array(shape_mask_image_2)

        # Note you may need to adjust the threshold according to your images' category
        low_threshold = 50  # 50#50 #100
        high_threshold = 150  # 150 #200
        canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

        canny_object = canny_image[y1_2.item():y2_2.item(), x1_2.item():x2_2.item(), :]
        color_object = color_image[y1_2.item():y2_2.item(), x1_2.item():x2_2.item(), :]
        shape_mask_image_2 = shape_mask_image_2[y1_2.item():y2_2.item(), x1_2.item():x2_2.item(), :]

        # When transforming, fill the top or left and right, it is enough, don't stretch, keep the aspect ratio
        length_height_ratio = (y2.item() - y1.item()) / (x2.item() - x1.item())
        canny_length_height_ratio = (y2_2.item() - y1_2.item()) / (x2_2.item() - x1_2.item())

        if length_height_ratio > canny_length_height_ratio:
            # fill top
            y_should = (x2_2.item() - x1_2.item()) * length_height_ratio - (y2_2.item() - y1_2.item())
            canny_object = np.concatenate([np.zeros((int(y_should), canny_object.shape[1], 3)), canny_object], axis=0)
            color_object = np.concatenate([np.zeros((int(y_should), color_object.shape[1], 3)), color_object], axis=0)
            shape_mask_image_2 = np.concatenate(
                [np.zeros((int(y_should), shape_mask_image_2.shape[1], 3)), shape_mask_image_2], axis=0)
        else:
            # fill left and right
            x_should = (y2_2.item() - y1_2.item()) / length_height_ratio - (x2_2.item() - x1_2.item())
            canny_object = np.concatenate([np.zeros((canny_object.shape[0], int(x_should / 2), 3)), canny_object,
                                           np.zeros((canny_object.shape[0], int(x_should / 2), 3))], axis=1)
            color_object = np.concatenate([np.zeros((color_object.shape[0], int(x_should / 2), 3)), color_object,
                                           np.zeros((color_object.shape[0], int(x_should / 2), 3))], axis=1)
            shape_mask_image_2 = np.concatenate(
                [np.zeros((shape_mask_image_2.shape[0], int(x_should / 2), 3)), shape_mask_image_2,
                 np.zeros((shape_mask_image_2.shape[0], int(x_should / 2), 3))], axis=1)

        canny_object = cv2.resize(canny_object, (x2.item() - x1.item(), y2.item() - y1.item()))
        color_object = cv2.resize(color_object, (x2.item() - x1.item(), y2.item() - y1.item()))

        canny_image = np.zeros_like(image)
        canny_image[y1.item():y2.item(), x1.item():x2.item(), :] = canny_object
        canny_image = _Image.fromarray(canny_image)

        color_image = np.zeros_like(image)
        color_image[y1.item():y2.item(), x1.item():x2.item(), :] = color_object
        color_image = _Image.fromarray(color_image)


        if mask_type == "draw mask":
            draw_mask = _Image.fromarray(cv2.resize(np.array(src_img), (512, 512)))
            mask_image = _Image.fromarray(np.uint8(draw_mask)).convert("RGB")
        elif src_msk_img is not None:
            mask_image =  _Image.fromarray(np.uint8(src_msk_img)).convert("RGB")
        new_image = pipe(
            text_prompt,
            num_inference_steps=50,  # 50,#50, #20,
            image=image,
            control_image=[canny_image, color_image],
            mask_image=mask_image,
            guess_mode=False,
            controlnet_conditioning_scale=[controlnet_scale_1, controlnet_scale_2],
            # You need to adjust this parameter according to the performance of generated images
            guidance_scale=15.5,
            # Larger will make the generated image more similar to the reference image, 15 pr more is recommended.
            strength=1.2
            # Must be larger than 1 to get enough denoise effect: https://www.bilibili.com/read/cv19739185/
        ).images[0]
        peak_memory = torch.cuda.max_memory_reserved(device=device) / (1024 ** 3)
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
        torch.cuda.empty_cache()
        end = time.time()
        print(f"Infer time: {end - start:.2f} s")
        print("type of new_image,", type(new_image))
        print("isinstance new_image, ", isinstance(new_image, _Image.Image))
        # return [new_image]
        return new_image
    except Exception as e:
        error_data_json = {
            "error_type": "Server Error!",
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
        }
        error_data = json.dumps(error_data_json)
        torch.cuda.empty_cache()
        raise gr.Error(error_data)


server = gr.Interface(
    fn=predict_v1,
    inputs=[gr.Image(tool ="sketch", type="pil", label="src image"), # maybe we can develop a function that users draw their own mask.
            gr.Image(type="pil", label="ref image"),
            gr.Textbox(label="mask object"),
            gr.Textbox(label="ref object"),
            gr.Radio(["auto mask", "draw mask"],value="auto mask",label="mask type"),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="controlnet_scale_1"),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.3, label="controlnet_scale_2"),
            gr.Image(type="pil", label="src mask image"),
            
            ],

    outputs=[gr.Image(label="generate image")],
    #outputs=gr.Textbox(label="test output"),
    title="image synthesis"
)
server.launch(server_name="0.0.0.0", server_port=port)
