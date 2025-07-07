import torch
import torch.nn as nn
from sousvide.control.networks.base_net import BaseNet
from torchvision.models import (
    vit_b_16,convnext_tiny,efficientnet_v2_s,
    ViT_B_16_Weights,ConvNeXt_Tiny_Weights,EfficientNet_V2_S_Weights
)
from transformers import AutoModel

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
    
class DINO(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="dino"):
        # Initialize the parent class
        super(DINO, self).__init__(inputs,outputs,network_type)

        # Load the DINO v2 model
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        model = AutoModel.from_pretrained("facebook/dinov2-base")

        # Class Variables        
        self.networks = model

    def forward(self, Xnn:dict[str,torch.Tensor]) -> dict[str, torch.Tensor]:

        # Unpack inputs
        xnn_vit = Xnn["rgb_image"]

        # Inference the ViT
        ynn_vit = self.networks(xnn_vit)

        # Extract the patches and class token
        pch = ynn_vit.last_hidden_state[:,1:,:].squeeze(0)
        cls = ynn_vit.last_hidden_state[:,0,:]

        # Reshape the patches back to (B,16,16,C)
        pch = pch.view(xnn_vit.shape[0],16,16,-1)

        # Create the output dictionary
        Ynn = {"patches": pch, "class_token": cls}
        
        return Ynn