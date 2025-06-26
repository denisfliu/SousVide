import torch
torch.set_float32_matmul_precision('high')

import os
import torch.nn as nn
import torch.nn.functional as F
import sousvide.utilities.feature_utilities as fu
import cv2
import numpy as np
import sousvide.control.networks.feature_extractors as fe
import figs.visualize.generate_videos as gv

from torch.utils.data import Dataset, DataLoader
from torchvision.models import squeezenet1_1

class HeatmapData(Dataset):
    def __init__(self,data_name:str):
        """
        Initialize the Heatmap Data.

        Args:
            data_path: Path to the heatmap data file.
            device: The device to use for loading the data.
        """
        # Define the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the number of data files in /data
        Ndf = 0
        for _ in os.listdir(data_name):
            Ndf += 1
        print(f"Number of data files found: {Ndf}")

        # Load the heatmap data
        for i in range(Ndf):
            data_path = os.path.join(data_name,"data" + str(i+1).zfill(3) + ".pt")
            data = torch.load(data_path, map_location=device)

            if i == 0:
                Xnn = data["Pch"]
                Ynn = data["Gss"]
            else:
                Xnn = torch.cat((Xnn, data["Pch"]), dim=0)
                Ynn = torch.cat((Ynn, data["Gss"]), dim=0)

        self.Xnn = Xnn
        self.Ynn = Ynn

    def __len__(self):
        return self.Xnn.shape[0]
    
    def __getitem__(self, idx):
        xnn = self.Xnn[idx]
        ynn = self.Ynn[idx]
        
        return xnn, ynn        

class SqNetMapData(Dataset):
    def __init__(self,data_name:str):
        """
        Initialize the Heatmap Data.

        Args:
            data_path: Path to the heatmap data file.
            device: The device to use for loading the data.
        """
        # Define the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the number of data files in /data
        Ndf = 0
        for _ in os.listdir(data_name):
            Ndf += 1
        print(f"Number of data files found: {Ndf}")

        # Load the heatmap data
        for i in range(Ndf):
            data_path = os.path.join(data_name,"data" + str(i+1).zfill(3) + ".pt")
            data = torch.load(data_path, map_location=device)

            if i == 0:
                Xnn = data["Img"]
                Ynn = data["Gss"]
            else:
                Xnn = torch.cat((Xnn, data["Img"]), dim=0)
                Ynn = torch.cat((Ynn, data["Gss"]), dim=0)

        self.Xnn = Xnn
        self.Ynn = Ynn

    def __len__(self):
        return self.Xnn.shape[0]
    
    def __getitem__(self, idx):
        xnn = self.Xnn[idx]
        ynn = self.Ynn[idx]
        
        return xnn, ynn        
    
class DistanceHeatmapRegressor(nn.Module):
    def __init__(self, input_dim:int=768):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, D, 16, 16) — e.g., patch embeddings reshaped into grid

        Returns:
            Tensor of shape (B, 16, 16) — predicted heatmap
        """
        x = x.permute(0, 3, 1, 2)
        out = self.net(x)

        return out.squeeze(1)

class SqNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            squeezenet1_1(),
            nn.Linear(1000,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )
    
    def forward(self, xnn):
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224) — e.g., input images

        Returns:
            Tensor of shape (B, 256) — feature representation
        """
        ynn = self.net(xnn)

        # Reshape to (B, 16, 16)
        ynn = ynn.view(ynn.size(0), 16, 16)  # Reshape to (B, 16, 16, 256)
        
        return ynn

def train_heatmap(model_name:str,data_name:str,epochs:int,
                  batch_size:int=64,lr:float=1e-3):

    # Set the device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model to device
    model = DistanceHeatmapRegressor()
    model.to(device)
    model.train()

    # Data loader
    dataset = HeatmapData(data_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        for xnn, ynn in dataloader:
            # Forward pass
            ypd = model(xnn)
            loss = criterion(ypd, ynn)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            # Print loss every 10 epochs or at the last epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

            # Save the trained model
            model_path = model_name+".pth"
            torch.save(model.state_dict(), model_path)


def train_sqnetmap(model_name:str,data_name:str,epochs:int,
                  batch_size:int=64,lr:float=1e-3):

    # Set the device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model to device
    model = SqNetRegressor()
    model.to(device)
    model.train()

    # Data loader
    dataset = SqNetMapData(data_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        for xnn, ynn in dataloader:
            # Forward pass
            ypd = model(xnn)
            loss = criterion(ypd, ynn)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            # Print loss every 10 epochs or at the last epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

            # Save the trained model
            model_path = model_name+".pth"
            torch.save(model.state_dict(), model_path)

def load_video_opencv(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    return frames, fps

def test_heatmap(model_name:str,test_type:str, tol:float):
    # If first two letters are "fe", use feature extractor
    if model_name[:2] == "fe":
        is_fe = True
    else:
        is_fe = False

    # Set the device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    if is_fe:
        model = DistanceHeatmapRegressor()
        vit = fe.DINOv2()
        vit.to(device)
        vit.eval()
    else:
        model = SqNetRegressor()

    # Load the trained model
    state_dict = torch.load(model_name + ".pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get image data
    if test_type == "test0":
        _,_,_,Iro,_,_ = fu.extract_rollout_data("features","button_app")
        fps = 20
    else:
        Iro,fps = load_video_opencv(test_type+".mp4")

    # Check if video folder exists
    if not os.path.exists("videos"):
        os.makedirs("videos")

    # Generate heatmaps
    Hmp = np.zeros((Iro.shape[0],Iro.shape[1],Iro.shape[2],3),dtype=np.uint8)
    for i in range(Iro.shape[0]):
        # Preprocess the image
        icr = fu.process_image(Iro[i]).to(device)

        # Generate heatmap
        if is_fe:
            pch,_ = vit(icr)
            gss = model(pch).squeeze(0).cpu().detach().numpy()
        else:
            gss = model(icr).squeeze(0).cpu().detach().numpy()

        Hmp[i,:,:,:] = fu.heatmap_overlay(gss, Iro[i],threshold=tol)
    
    video_path = os.path.join("videos", model_name + "_" + test_type + ".mp4")
    gv.images_to_mp4(Hmp,video_path,fps=fps)