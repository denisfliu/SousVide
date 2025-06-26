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
from scipy.ndimage import zoom

transform = A.Compose([                                             # Image transformation pipeline
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])            

def process_image(image:np.ndarray) -> torch.Tensor:
    image = transform(image=image)["image"]
    image = image.unsqueeze(0)  # Add batch dimension

    return image

def compute_similarity(target:torch.Tensor, patches:torch.Tensor):

    Nr,Nc = patches.shape[0],patches.shape[1]
    cSm = torch.zeros((Nr,Nc))
    for i in range(Nr):
        for j in range(Nc):
            # Compute the similarity between the target and each patch
            cSm[i,j] = torch.cosine_similarity(target, patches[i,j], dim=0)

    return cSm

def get_patch_indices(cSm:torch.Tensor, threshold:float=0.8, max_patches:int=1):
    """
    Get the indices of the patches that are above a certain similarity threshold.
    
    Args:
        cSm: cosine similarity matrix
        threshold: similarity threshold
        max_patches: maximum number of patches to return

    Returns:
        indices: list of tuples (i, j) for patches above the threshold
    """
    flat = cSm.flatten()
    mask = flat >= threshold
    values = flat[mask]

    if values.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)

    # Get original flat indices above threshold
    flat_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # If too many, take top-k
    if values.numel() > max_patches:
        topk = torch.topk(values, max_patches)
        flat_indices = flat_indices[topk.indices]

    # Convert to 2D (row, col)
    rows, cols = torch.div(flat_indices, cSm.shape[1], rounding_mode='floor'), flat_indices % cSm.shape[1]
    
    return torch.stack([rows, cols], dim=1)

def similarity_overlay(idxs:torch.Tensor,image:np.ndarray,
                       Nr:int=16,Nc:int=16,colormap='seismic', alpha=0.5):
    """
    Overlay a 16x16 heatmap on a 360x640 RGB image.
    
    Args:
        idxs: indices of the patches to overlay
        image: RGB image (H, W, 3)
        Nr: number of rows in the heatmap
        Nc: number of columns in the heatmap
        colormap: colormap to use for the overlay
    Returns:
        overlayed RGB image (uint8)
    """

    # Create a blank heatmap
    heatmap = torch.zeros((Nr, Nc))
    for idx in idxs:
        i, j = idx[0].item(), idx[1].item()
        heatmap[i, j] = 1.0

    # Resize heatmap to match image size
    heatmap_np = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 16)
    upsampled = torch.nn.functional.interpolate(
        heatmap_np, size=image.shape[:2], mode='nearest'
    ).squeeze().numpy()  # (H, W)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(upsampled)[:, :, :3]  # drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay

def pca_overlay(pca:np.ndarray,threshold:float=0.5):
    
    # Check if images are float or uint8
    if pca.dtype == np.uint8:
        pca = pca.astype(np.float32) / 255.0

    # Compute the RGB channels
    pc1 = pca[:, :, 0]
    pc2 = pca[:, :, 1]
    pc3 = pca[:, :, 2]

    foreground_mask = pc1 < threshold

    r = (pc1-pc1.min()) / (pc1.max()-pc1.min())
    g = (pc2-pc2.min()) / (pc2.max()-pc2.min())
    b = (pc3-pc3.min()) / (pc3.max()-pc3.min())

    # Combine the channels
    image = np.zeros((pca.shape[0], pca.shape[1], 3), dtype=np.uint8)

    image[:,:,0] = (r * 255).astype(np.uint8)
    image[:,:,1] = (g * 255).astype(np.uint8)
    image[:,:,2] = (b * 255).astype(np.uint8)

    image[foreground_mask] = (0, 0, 0)  # Set foreground to black
    return image

def heatmap_overlay(heatmap:np.ndarray,image:np.ndarray,
                    colormap='plasma', alpha=0.5,threshold:float|None=None):
    
    # Check if images are float or uint8
    if heatmap.dtype == np.uint8:
        heatmap = heatmap.astype(np.float32) / 255.0
    elif heatmap.dtype == torch.float32:
        heatmap = heatmap.cpu().numpy()

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == torch.float32:
        image = image.cpu().numpy()

    # Apply threshold if provided
    if threshold is not None:
        heatmap[heatmap < threshold] = 0
        
    # Get the scale factors
    H,W = image.shape[:2]
    h,w, = heatmap.shape
    zoom_y = H / h
    zoom_x = W / w

    # Resize heatmap to match image size
    heatmap_upsampled = zoom(heatmap, (zoom_y, zoom_x), order=0)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_upsampled)[:, :, :3]

    # Blend with original image
    overlay = alpha*image + (1 - alpha)*heatmap_colored
    overlay = (overlay * 255).astype(np.uint8)

    return overlay

def extract_rollout_data(cohort:str,course:str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
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
    Iro = imgs["rgb"]

    Tc2b = np.array(traj["frame"]["camera_to_body_transform"])
    fx,fy = traj["frame"]["camera"]["fx"],traj["frame"]["camera"]["fy"]
    cx,cy = traj["frame"]["camera"]["cx"],traj["frame"]["camera"]["cy"]
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])

    return Tro,Xro,Uro,Iro,Tc2b,K
