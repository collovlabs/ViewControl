

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from diffusers.utils import load_image
import argparse



def main(input_path, output_path):

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    prompt = ""
    image = load_image(input_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    with open(output_path, "w") as f:
        f.write(generated_text)

    print("image from {} captioned as {}".format(input_path, generated_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./input_images_path")
    parser.add_argument("--output_path", type=str, default="./output_caption_path")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
