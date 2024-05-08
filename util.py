import argparse
from PIL import Image
from lang_sam import LangSAM 
import numpy as np
import argparse


def segmentation(image, text, output_path):

    model = LangSAM(sam_type="vit_h")

    def predict(image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        if isinstance(image_path, str):
            image_pil = Image.open(image_path).convert("RGB")
        else:
            # bug here, need to be improved
            image_pil = image_path
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        image_array = np.asarray(image_pil.convert("RGBA"))

        output_image = np.zeros_like(image_array)
        output_image[:,:,3] = 255
        output_image[:,:,0:3] = image_array[:,:,0:3]

        for i in range(len(masks)):
            mask = masks[i]
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 4, axis=2)
            mask = mask.astype(np.uint8)
            mask = mask * 255
            output_image = np.where(mask > 0, output_image, 0)

        output_image = Image.fromarray(np.uint8(output_image)).convert("RGBA")

        output_image.save(output_path)

        
    predict(image, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./input_images_path")
    parser.add_argument("--prompt", type=str, default="sofa")
    parser.add_argument("--output_path", type=str, default="./output_images_path")

    args = parser.parse_args()
    segmentation(args.input_path, args.prompt, args.output_path)