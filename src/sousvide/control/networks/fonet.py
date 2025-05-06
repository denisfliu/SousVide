import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class FONet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="fonet"):


        # Initialize the parent class
        super(FONet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]

        # Unpack network configs from parent
        prev_size,_,output_size = self.get_io_sizes()

        # Populate the network
        hidden_sizes = layers["hidden_sizes"]

        networks = []
        for layer_size in hidden_sizes:
            networks.append(nn.Linear(prev_size, layer_size))
            networks.append(nn.ReLU())
            networks.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        networks.append(nn.Linear(prev_size, output_size))

        # Class Variables
        self.networks = nn.Sequential(*networks)

    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:    Current state input.

        Returns:
            ynn:    Output tensor.
        """

        # Forward pass
        ynn = self.networks(xnn)

        return ynn