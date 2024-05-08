from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline,UNet2DConditionModel,DDIMScheduler,StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import cv2
import os
import subprocess
from PIL import Image
from lang_sam import LangSAM # cd ./lang-segment-anything; pip3 install -e . #cd ./GroundingDINO; pip install -e . # https://github.com/IDEA-Research/GroundingDINO/issues/8 for NameError: name '_C' is not defined
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
import numpy as np
import argparse


def bbox(image="/path/img.png", text="sofa"):

    model = LangSAM(sam_type="vit_h")

    def draw_image(image, masks, boxes, labels, alpha=0.4):
        image = torch.from_numpy(image).permute(2, 0, 1)
        if len(boxes) > 0:
            image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
        if len(masks) > 0:
            image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
        return image.numpy().transpose(1, 2, 0)


    def predict(image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        if isinstance(image_path, str):
            image_pil = Image.open(image_path).convert("RGB")
        else:
            # bug here, need to be improved
            image_pil = image_path
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        image_array = np.asarray(image_pil)
        image = draw_image(image_array, masks, boxes, labels)
        image = Image.fromarray(np.uint8(image)).convert("RGB")

        mask_image1 = draw_image(np.zeros_like(image_array), masks, [], [], alpha=1.0)
        mask_image1 = Image.fromarray(np.uint8(mask_image1)).convert("RGB")

        boxes_mask = torch.zeros_like(masks)
        new_boxes=[]

        # if len(boxes)>1:
        #     boxes[0] = boxes[1]

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1)-1, int(y1)-1, int(x2)+1, int(y2)+1
            new_boxes.append([x1, y1, x2, y2])
            boxes_mask[:, y1:y2, x1:x2] = 1


        mask_image2 = draw_image(np.zeros_like(image_array), boxes_mask, [], [], alpha=1.0)
        mask_image2 = Image.fromarray(np.uint8(mask_image2)).convert("RGB")
        boxes = torch.tensor(new_boxes)
        shape_mask_image = draw_image(np.zeros_like(image_array), masks, [], [], alpha=1.0)
        shape_mask_image = Image.fromarray(np.uint8(shape_mask_image)).convert("RGB")
        return mask_image2, boxes, image, shape_mask_image
        

    mask_image, boxes, image, shape_mask_image = predict(image, text)
    image.save("./tmp/tmp.png")
    mask_image.save("./tmp/mask_image.png")
    return mask_image, boxes, shape_mask_image

def train_dreambooth(pipe):
    if dreambooth:
        # IF YOU NEED TO PERSONALIZE THE MODEL
        process =  subprocess.Popen([
            'python', './diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py', 
            '--pretrained_model_name_or_path=' + os.environ['MODEL_NAME'],
            '--instance_data_dir=' + os.environ['INSTANCE_DIR'],
            '--output_dir=' + os.environ['OUTPUT_DIR'],
            '--instance_prompt=' + os.environ['TEXT_PROMPT'],
            # '--class_prompt=' + os.environ['CLASS_PROMPT'],
            '--resolution=512',
            '--train_batch_size=1',
            '--gradient_accumulation_steps=1',
            '--checkpointing_steps=1800',
            '--learning_rate=2e-6',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0',
            '--max_train_steps=1800',
            '--mixed_precision=fp16',
        ], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        

        # ABONDED, THIS IS FOR LORA
        # subprocess.run([
        #     'accelerate', 'launch', './diffusers/examples/dreambooth/train_dreambooth_lora.py',
        #     '--pretrained_model_name_or_path=' + os.environ['MODEL_NAME'],
        #     '--instance_data_dir=' + os.environ['INSTANCE_DIR'],
        #     '--output_dir=' + lora_model_path,
        #     '--instance_prompt=' + text_prompt,
        #     '--resolution=256',
        #     '--train_batch_size=1',
        #     '--gradient_accumulation_steps=1',
        #     '--checkpointing_steps=100',
        #     '--learning_rate=1e-4',
        #     '--lr_scheduler=constant',
        #     '--lr_warmup_steps=0',
        #     '--max_train_steps=500',
        #     '--validation_prompt=' + text_prompt,
        #     '--validation_epochs=50',
        #     '--seed=0'
        # ])

        print('dreambooth training is time consuming, please wait for a while, or you can use the pretrained model')

        for line in process.stdout:
            print(line, end='')

        process.wait()

        print('finished finetune')
        unet = UNet2DConditionModel.from_pretrained(os.environ['OUTPUT_DIR']+'/unet', torch_dtype=torch.float16)
        pipe.unet = unet


def image_synthesis(path_src_img, path_ref_img, text_prompt, save_path, mask_obj_name, ref_obj_name):
    image = load_image(path_src_img) 
    canny_image = load_image(path_ref_img) 

    pipe.to(device)
    generator = torch.manual_seed(12345)

    torch.cuda.empty_cache()

    # Note that the image size should be as close as possible to the size of the original image, otherwise the generated image will be blurred.
    image = Image.fromarray(cv2.resize(np.array(image), (512,512)))
    canny_image = Image.fromarray(cv2.resize(np.array(canny_image), (512,512)))

    mask_image, boxes, shape_mask_image= bbox(image, mask_obj_name)
    canny_image.save("./tmp/tmp_ori_canny_image.png")

    # get the bbox number of the mask, when then is only one box
    x1, y1, x2, y2 = boxes[0]
    print(" boxes[0]", boxes[0])


    mask_image_2, boxes_2, shape_mask_image_2 = bbox(canny_image, ref_obj_name)
    x1_2, y1_2, x2_2, y2_2 = boxes_2[0]

    # adjust the canny image, so that the bbox of the mask is the same as the bbox of the canny image
    canny_image = np.array(canny_image)
    color_image = canny_image

    shape_mask_image = np.array(shape_mask_image)
    shape_mask_image_2 = np.array(shape_mask_image_2)

    # Note you may need to adjust the threshold according to your images' category
    low_threshold = 50#50#50 #100
    high_threshold = 150#150 #200
    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)


    canny_object = canny_image[y1_2.item():y2_2.item(),x1_2.item():x2_2.item(),:]
    color_object = color_image[y1_2.item():y2_2.item(),x1_2.item():x2_2.item(),:]
    shape_mask_image_2 = shape_mask_image_2[y1_2.item():y2_2.item(),x1_2.item():x2_2.item(),:]

    # When transforming, fill the top or left and right, it is enough, don't stretch, keep the aspect ratio
    length_height_ratio = (y2.item()-y1.item())/(x2.item()-x1.item())
    canny_length_height_ratio = (y2_2.item()-y1_2.item())/(x2_2.item()-x1_2.item())

    if length_height_ratio > canny_length_height_ratio:
        # fill top 
        y_should = (x2_2.item()-x1_2.item())*length_height_ratio - (y2_2.item()-y1_2.item())
        canny_object = np.concatenate([np.zeros((int(y_should),canny_object.shape[1],3)),canny_object],axis=0)
        color_object = np.concatenate([np.zeros((int(y_should),color_object.shape[1],3)),color_object],axis=0)
        shape_mask_image_2 = np.concatenate([np.zeros((int(y_should),shape_mask_image_2.shape[1],3)),shape_mask_image_2],axis=0)
    else:
        # fill left and right
        x_should = (y2_2.item()-y1_2.item())/length_height_ratio - (x2_2.item()-x1_2.item())
        canny_object = np.concatenate([np.zeros((canny_object.shape[0],int(x_should/2),3)),canny_object,np.zeros((canny_object.shape[0],int(x_should/2),3))],axis=1)
        color_object = np.concatenate([np.zeros((color_object.shape[0],int(x_should/2),3)),color_object,np.zeros((color_object.shape[0],int(x_should/2),3))],axis=1)
        shape_mask_image_2 = np.concatenate([np.zeros((shape_mask_image_2.shape[0],int(x_should/2),3)),shape_mask_image_2,np.zeros((shape_mask_image_2.shape[0],int(x_should/2),3))],axis=1)

    canny_object = cv2.resize(canny_object,(x2.item()-x1.item(),y2.item()-y1.item()))
    color_object = cv2.resize(color_object,(x2.item()-x1.item(),y2.item()-y1.item()))
    shape_mask_image_2 = cv2.resize(shape_mask_image_2,(x2.item()-x1.item(),y2.item()-y1.item()))


    canny_image = np.zeros_like(image)
    canny_image[y1.item():y2.item(),x1.item():x2.item(),:] = canny_object
    canny_image = Image.fromarray(canny_image)

    color_image = np.zeros_like(image)
    color_image[y1.item():y2.item(),x1.item():x2.item(),:] = color_object
    color_image = Image.fromarray(color_image)

    shape_image = np.zeros_like(image)
    shape_image[y1.item():y2.item(),x1.item():x2.item(),:] = shape_mask_image_2


    shape_mask_image_1_ori = shape_mask_image
    shape_mask_image_1_ori = Image.fromarray(np.array(shape_mask_image_1_ori))
    shape_mask_image_2_ori = shape_image
    shape_mask_image_2_ori = Image.fromarray(np.array(shape_mask_image_2_ori))


    # STAGE 1: PURE INPAINT
    mid_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to(device)
    mid_image = mid_pipe(
        prompt= "Remove anything, empty, clean", #"floor and wall",
        negative_prompt = text_prompt,
        num_inference_steps=20,
        generator=generator,
        image=image,
        mask_image= mask_image #shape_mask_image
    ).images[0]
    mid_image.save("./tmp/tmp_mid_image2.png")
    ori_mid_image = mid_image
    mid_image = cv2.resize(np.array(mid_image), (color_image.size[0],color_image.size[1]))
   
    color_image = np.array(color_image)
  
    # shape_mask_image_2_ori is a [512,512,3] image
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            if shape_image[i,j,0] == 0 and shape_image[i,j,1] == 0 and shape_image[i,j,2] == 0:
               
                color_image[i,j,:] =[mid_image[i,j,:][0],mid_image[i,j,:][1],mid_image[i,j,:][2]]

    color_image = Image.fromarray(color_image)

    mask_image.save("./tmp/tmp_mask_image.png")

    ori_mid_image.save("./tmp/tmp_mid_image_0927.png")
    shape_mask_image_2_ori.save("./tmp/tmp_shape_mask_image_2_ori_0927.png")

    canny_image.save("./tmp/tmp_canny_image.png")
    color_image.save("./tmp/tmp_color_image.png")

    if TWO_STAGE:
        new_image = pipe(
            text_prompt,
            num_inference_steps=50, #50,#50, #20,
            generator=generator,
            image= ori_mid_image,#image,
            control_image=[canny_image,color_image],
            mask_image=shape_mask_image_2_ori, #mask_image,
            guess_mode = False,
            controlnet_conditioning_scale =[0.2,0.1], 
            guidance_scale=15.5,
            strength=1.2
        ).images[0]
    else:
        new_image = pipe(
            text_prompt,
            num_inference_steps=50, #50,#50, #20,
            generator=generator,
            image= image,
            control_image=[canny_image,color_image],
            mask_image=mask_image,
            guess_mode = False,
            controlnet_conditioning_scale =[0.3,0.1], # You need to adjust this parameter according to the performance of generated images
            guidance_scale=15.5,# Larger will make the generated image more similar to the reference image, 15 pr more is recommended.
            strength=1.2 # Must be larger than 1 to get enough denoise effect: https://www.bilibili.com/read/cv19739185/
        ).images[0]


    new_image.save(save_path)

    return new_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_src_img', type=str, default="./imgs/sofa_set/sofa_bg_f2.png")
    parser.add_argument('--path_ref_img', type=str, default="./imgs/synthesized_imgs/sofa_1_a/0_20.png")
    parser.add_argument('--text_prompt', type=str, default="sofa_1_a")
    parser.add_argument('--save_path', type=str, default="./tmp/tmp_result.png")
    parser.add_argument('--mask_obj_name', type=str, default="sofa")
    parser.add_argument('--ref_obj_name', type=str, default="sofa")

    parser.add_argument('--dreambooth', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()

    text_prompt = "A sks " + args.text_prompt
    NAME = args.text_prompt
    os.environ['MODEL_NAME'] = "runwayml/stable-diffusion-inpainting"
    os.environ['INSTANCE_DIR'] = "./imgs/synthesized_imgs/" + NAME + "/"
    os.environ['OUTPUT_DIR'] = "./models/"+NAME
    os.environ['TEXT_PROMPT'] = "a photo of sks " + NAME # Some times you may need to change this to get better results, for example, it is a couch with wooden legs." #, it is a couch with a gray fabric covering on it" #"pure white sks teapot"#"green sks A_green_couch" #"One blue sks A_blue_tea_kettle on an indoor floor"
    # os.environ['CLASS_PROMPT'] = NAME.split("_")[0]

    dreambooth = args.dreambooth
    device = args.device
    TWO_STAGE = False #True

    controlnet = [ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
                ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16),
                ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None 
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if dreambooth:
        train_dreambooth(pipe)
    image_synthesis(args.path_src_img, args.path_ref_img, text_prompt, args.save_path, args.mask_obj_name, args.ref_obj_name)

