import torch
import sousvide.control.network_helper as nh

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

        Args:
            inputs:         Inputs config.
            outputs:        Outputs config.
            layers:         Layers config.
            dropout:        Dropout rate.
            network_type:   Type of network.

        Variables:
            network_type:   Type of network.
            input_indices:  Indices of the input.
            fpass_indices:  Indices of the forward-pass output.
            label_indices:  Indices of the label output.
            networks:       Network layers.
        """

        # Initialize the parent class
        super(SVNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        Nsq = layers["sqnet_size"]
        hidden_sizes = layers["hidden_sizes"]
        idx_cmd = layers["cmd_aug_layer"]

        # Unpack network configs from parent
        input_sizes,_,output_sizes = self.get_io_sizes(expanded=True)
        Ncr1,Ncr2,NhL = input_sizes[1],input_sizes[2],input_sizes[3]
        output_size = output_sizes[0]

        # Populate the network
        prev_size = Ncr1+Nsq

        mlp0 = []
        for layer_size in hidden_sizes[:idx_cmd]:
            mlp0.append(nn.Linear(prev_size, layer_size))
            mlp0.append(nn.ReLU())
            mlp0.append(nn.Dropout(dropout))

            prev_size = layer_size

        prev_size += Ncr2 + NhL

        mlp1 = []
        for layer_size in hidden_sizes[idx_cmd:]:
            mlp1.append(nn.Linear(prev_size, layer_size))
            mlp1.append(nn.ReLU())
            mlp1.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        mlp1.append(nn.Linear(prev_size, output_size))

        # Populate the network
        networks = nn.ModuleDict({
            "feat": squeezenet1_1(),
            "mlp0": nn.Sequential(*mlp0),
            "mlp1": nn.Sequential(*mlp1)
        })

        # Class Variables
        self.networks = networks
    
    def forward(self,
                xnn_img:torch.Tensor,
                xnn_cr1:torch.Tensor,
                xnn_cr2:torch.Tensor,
                xnn_hL:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_img: Image input.
            xnn_cr1: Current input (first instance).
            xnn_cr2: Current input (second instance).
            xnn_hL: History Network input.

        Returns:
            ynn:    Output tensor.
        """

        # Feature extraction
        znn = self.networks["feat"](xnn_img)

        # Image MLP
        znn = torch.cat([znn,xnn_cr1],dim=1)
        znn = self.networks["mlp0"](znn)

        # Command MLP
        znn = torch.cat([znn,xnn_cr2,xnn_hL],dim=1)
        ynn = self.networks["mlp1"](znn)     

        return ynn