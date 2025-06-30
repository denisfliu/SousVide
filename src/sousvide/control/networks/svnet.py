import torch

from torch import nn
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet

class SVNet(BaseNet):
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
        super(SVNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs
        io_sizes = self.get_io_sizes()
        dropout = layers["dropout"]
        Nsq = layers["sqnet_size"]
        hidden_sizes = layers["hidden_sizes"]
        idx_cmd = layers["cmd_aug_layer"]

        # Populate the network
        prev_size = io_sizes["xdp"]["current1"]+Nsq

        mlp0 = []
        for layer_size in hidden_sizes[:idx_cmd]:
            mlp0.append(nn.Linear(prev_size, layer_size))
            mlp0.append(nn.ReLU())
            mlp0.append(nn.Dropout(dropout))

            prev_size = layer_size

        prev_size += io_sizes["xdp"]["current2"] + io_sizes["xdp"]["feature_vector"]

        mlp1 = []
        for layer_size in hidden_sizes[idx_cmd:]:
            mlp1.append(nn.Linear(prev_size, layer_size))
            mlp1.append(nn.ReLU())
            mlp1.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        mlp1.append(nn.Linear(prev_size, io_sizes["ypd"]["command"]))

        # Populate the network
        networks = nn.ModuleDict({
            "feat": squeezenet1_1(),
            "mlp0": nn.Sequential(*mlp0),
            "mlp1": nn.Sequential(*mlp1)
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
        xnn_img,xnn_hL = Xnn["rgb_image"],Xnn["feature_vector"]
        xnn_cr1,xnn_cr2 = Xnn["current1"],Xnn["current2"]

        # Feature extraction
        ynn_ft = self.networks["feat"](xnn_img)

        # Image MLP
        xnn_im = torch.cat([ynn_ft,xnn_cr1],dim=1)
        ynn_im = self.networks["mlp0"](xnn_im)

        # Command MLP
        xnn_cm = torch.cat([ynn_im,xnn_cr2,xnn_hL],dim=1)
        ynn_cm = self.networks["mlp1"](xnn_cm)     

        # Create the output dictionary
        Ynn = {"command": ynn_cm}

        return Ynn