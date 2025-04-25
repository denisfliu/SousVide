import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16,convnext_tiny,efficientnet_v2_s,
    ViT_B_16_Weights,ConvNeXt_Tiny_Weights,EfficientNet_V2_S_Weights
)

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
        x = x[:, 1:, :]

        return x
    
class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def forward(self, x):
        x = self.vit.get_intermediate_layers(x,12)[-1]

        return x