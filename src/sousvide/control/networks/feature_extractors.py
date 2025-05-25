import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16,convnext_tiny,efficientnet_v2_s,
    ViT_B_16_Weights,ConvNeXt_Tiny_Weights,EfficientNet_V2_S_Weights
)
import os
from transformers import AutoProcessor, AutoModel

class VitB16(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Run the input through the transformer encoder
        x = self.vit.encoder(x)

        # Extract the patches
        ynn = x[:, 1:, :]
        cls = x[:, 0, :]

        ynn = ynn.squeeze(0)
        cls = cls.squeeze(0)

        return ynn,cls
    
class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()

        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        model_name = "facebook/dinov2-base"  # Choose your desired DINO v2 variant
        # processor = AutoProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)
        self.vit = backbone

    def forward(self, xnn) -> tuple[torch.Tensor,torch.Tensor]:
        outputs = self.vit(xnn)

        ynn = outputs.last_hidden_state[:,1:,:]
        cls = outputs.last_hidden_state[:,0,:]

        return ynn,cls