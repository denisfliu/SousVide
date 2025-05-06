# Import the relevant modules
import os
import torch
import albumentations as A
import figs.visualize.generate_videos as gv
import sousvide.control.networks.feature_extractors as fe
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

from sklearn.decomposition import PCA
from albumentations.pytorch import ToTensorV2
from sousvide.control.pilot import Pilot

transform = A.Compose([                                             # Image transformation pipeline
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])            
process_image = lambda x: transform(image=x)["image"]


def overlay_heatmap_on_image(heatmap, image, colormap='viridis', alpha=0.5):
    """
    Overlay a 16x16 heatmap on a 360x640 RGB image.
    
    Args:
        heatmap: (16, 16) array, float or uint8
        image: (360, 640, 3) RGB image (uint8)
        colormap: name of a matplotlib colormap
        alpha: blending factor [0 = only image, 1 = only heatmap]

    Returns:
        overlayed RGB image (uint8)
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
        
    # Normalize heatmap to [0, 1]
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay

def extract_rollout_data(cohort:str,course:str):
    workspace = os.path.join("../cohorts",cohort,"rollout_data",course)

    # Load the Rollout Data
    trajs_path = os.path.join(workspace,"trajectories")
    imgs_path = os.path.join(workspace,"images")

    # Extract the image and trajectory files
    trajs_files = [os.path.join(trajs_path,f) for f in os.listdir(trajs_path) if f.endswith('.pt')]
    imgs_files = [os.path.join(imgs_path,f) for f in os.listdir(imgs_path) if f.endswith('.pt')]

    trajs_files.sort()
    imgs_files.sort()

    trajs_file = trajs_files[:1]
    imgs_file = imgs_files[:1]

    # Load the data
    traj = torch.load(trajs_file[0])[0]
    imgs = torch.load(imgs_file[0])[0]

    Tro,Xro,Uro = traj["Tro"],traj["Xro"],traj["Uro"]
    Iro = imgs["images"]

    return Tro,Xro,Uro,Iro
