import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
import random

name = "log_dinov2_mlp_1e-5_tmp"
gpuid = 1
from transformers import AutoImageProcessor, Dinov2Model

import logging
import sys
logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    handlers=[logging.FileHandler("{}.log".format(name)),
                        logging.StreamHandler(sys.stdout) ] )
device ="cuda:{}".format(gpuid)
model_save_path = 'best_model_{}.pth'.format(name)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
vit_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
bs = 128 #100
lr = 1e-5
data_folder = "./imgs/pose_estimation_train_dataset"  

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test_split=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_split = test_split
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []

        fur_dir_list = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for fur_dir in fur_dir_list:
            if "dreambooth" in fur_dir and "old" not in  fur_dir:
                fur_path = os.path.join(self.root_dir, fur_dir)
                deg_dir_list = [d for d in os.listdir(fur_path) if os.path.isdir(fur_path )]
                
                for deg_dir in deg_dir_list :
                    deg_path = os.path.join(fur_path, deg_dir)
                    files = [f for f in os.listdir(deg_path) if f.endswith('.png')]
                    for file in files:
                        file_path = os.path.join(deg_path, file)
                        r, t = file.split('_')
                        label = [float(r), float(t[:-4])]  # 移除文件名中的".png"后缀
                        data.append(file_path)
                        labels.append(label)
        
        zipped = list(zip(data, labels))
        random.shuffle(zipped)
        data, labels = zip(*zipped)

        split_index = int(len(data) * (1 - self.test_split))
        if self.train:
            data = data[:split_index]
            labels = labels[:split_index]
        else:
            data = data[split_index:]
            labels = labels[split_index:]

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'][0]
        inputs['label'] = label
        return inputs


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


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = CustomDataset(data_folder, transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=32)
test_dataset = CustomDataset(data_folder, transform=transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=32)


model = RegressionModel().float().to(device)
model = model.to(device)


criterion = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer = optim.AdamW(model.parameters(), lr=lr)


round_list =np.array (list(range(-160,170,10)))
def discretize(outputs):
    outputs = outputs.numpy()
    round_cand = round_list
    for i in range(len(outputs.shape)):
        round_cand = np.expand_dims(round_cand, 0)
    outputs = np.expand_dims(outputs, -1)
    diff = np.abs(outputs-round_cand)
    pos = np.expand_dims(np.argmin(diff, axis = -1), axis = -1)
    res = round_list[pos].squeeze()
    return torch.tensor(res)

lowest_loss = float('inf')


num_epochs = 500
discrete_val = []
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch['pixel_values'], batch['label']
        labels = torch.stack(labels, dim=-1)

        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # from thop import profile
        # flops, params = profile(model, (inputs,))
        # print('flops: ', flops, 'params: ', params)
        
        loss = criterion(outputs, labels)
        loss_mae = criterion_mae(outputs.detach().cpu(), labels.detach().cpu())

        loss.backward()
        optimizer.step()
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], train MSE: {loss.item():.2f}, mae: {loss_mae.item():.2f}, RMSE: {np.sqrt(loss.item()):.2f}')
        
        outputs_round =discretize(outputs.detach().cpu()) 
        loss = criterion(outputs_round.detach().cpu(), labels.detach().cpu())
        loss_mae = criterion_mae(outputs_round.detach().cpu(), labels.detach().cpu())
        logging.info(f'Epoch [{epoch+1}/{num_epochs}],     r MSE: {loss.item():.2f}, mae: {loss_mae.item():.2f},  RMSE: {np.sqrt(loss.item()):.2f}')

    # After each epoch, the model will be evaluated on the test set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_loss_mae = 0

        r_total_loss = 0
        r_total_loss_mae = 0
        
        for batch in test_loader:
            inputs, labels = batch['pixel_values'], batch['label']
            labels = torch.stack(labels, dim=-1)
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss_mae = criterion_mae(outputs.detach().cpu(), labels.detach().cpu())

            
            total_loss += loss.item()
            total_loss_mae += loss_mae.item()

            
            outputs_round =discretize(outputs.detach().cpu()) 
            r_loss = criterion(outputs_round.detach().cpu(), labels.detach().cpu())
            r_loss_mae = criterion_mae(outputs_round.detach().cpu(), labels.detach().cpu())
    

            r_total_loss += r_loss.item()
            r_total_loss_mae += r_loss_mae.item()

            
            
        average_loss = total_loss / len(test_loader)
        average_loss_mae = total_loss_mae / len(test_loader)

        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Test MSE: {average_loss:.2f}, mae: {average_loss_mae:.2f}, RMSE: {np.sqrt(average_loss):.2f}')
    
        average_loss = r_total_loss / len(test_loader)
        average_loss_mae = r_total_loss_mae / len(test_loader)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}],    r MSE: {average_loss:.2f}, mae: {average_loss_mae:.2f},  RMSE: {np.sqrt(average_loss):.2f}')

    # If the test loss of the current model is lower, save the current model
    if average_loss < lowest_loss:
        lowest_loss = average_loss
        torch.save(model.state_dict(), model_save_path)
        logging.info(f'Saved model with lowest test loss {lowest_loss}: {model_save_path}')

