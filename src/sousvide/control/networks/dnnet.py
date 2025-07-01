import torch

from torch import nn
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet

class DNNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="svnet"):
        """
        Initialize a SousVide Command Network.

        The network takes images, current states and histrory feature vector to
        output a command vector. The network uses a SqueezeNet backbone to extract
        image features.

        """

        # Initialize the parent class
        super(DNNet, self).__init__(inputs,outputs,network_type)

        # # Unpack network configs
        # io_sizes = self.get_io_sizes()

        # Populate the network
        hmap = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        prev_size = 256 + 8
        hidden_sizes = [128, 64]
        mlp = []
        for layer_size in hidden_sizes:
            mlp.append(nn.Linear(prev_size, layer_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(0.1))

            prev_size = layer_size
        mlp.append(nn.Linear(prev_size, 4))

        # Populate the network
        networks = nn.ModuleDict({
            "hmap": hmap,
            "mlp": nn.Sequential(*mlp),
        })

        # Class Variables
        self.networks = networks
    
    def forward(self,Xnn:dict[str,torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            Xnn:    Input dictionary.

        Returns:
            Ynn:    Output dictionary.
        """

        # Unpack inputs
        xnn_ft,xnn_hL = Xnn["patches"],Xnn["feature_vector"]

        # Heatmap extraction
        xnn_ft =  xnn_ft.permute(0, 3, 1, 2)
        ynn_hm = self.networks["hmap"](xnn_ft)
        ynn_hm = ynn_hm.squeeze(1).flatten(1,2)  # Flatten to (B, H*W)

        # Image MLP
        xnn_cm = torch.cat([ynn_hm,xnn_hL],dim=1)
        ynn_cm = self.networks["mlp"](xnn_cm)
   
        # Create the output dictionary
        Ynn = {"command": ynn_cm}

        return Ynn