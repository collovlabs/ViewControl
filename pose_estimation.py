import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
vit_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.vit = vit_model
        # self.fc1 = nn.Linear(768, 768)
        # self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, 128)
        self.fc4 = nn.Linear(128, 2)

        
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.vit(x)
        sequence_output = outputs[0]
        x = sequence_output[:, 0, :] #[B,768]
        
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main(input_path, output_path, model_path, device="cuda"):
    # inference
    # input a image, and predict R and T

    input_image = Image.open(input_path).convert('RGB')
    input_image = processor(images=input_image, return_tensors="pt")
    input_image = input_image['pixel_values'].to(device)
    
    model = RegressionModel().float().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        prediction = model(input_image)[0]
        predicted_R, predicted_T = prediction[0], prediction[1]
        # round to integer and write to output_path. You may change this part to round to the integer that can be divided by 10
        predicted_R = round(predicted_R.item())
        predicted_T = round(predicted_T.item())
        with open(output_path, "w") as f:
            f.write(f"{predicted_R} {predicted_T}")
        logging.info(f'Predicted R: {predicted_R}, Predicted T: {predicted_T}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./input_images_path")
    parser.add_argument("--output_path", type=str, default="./output_pose_path")
    parser.add_argument("--model_path", type=str, default="./model_path")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.model_path, args.device)