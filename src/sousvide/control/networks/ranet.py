import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class RANet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="ranet"):
        """
        Initialize a Rapid Adaptation Network model.

        The network takes in a current input and a history input into
        a transformer and outputs a motor command.
        """

        # Initialize the parent class
        super(RANet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]

        # Unpack network configs from parent
        prev_size,hisLat_size,output_size = self.get_io_sizes()

        # Populate the network
        hidden_sizes = layers["hidden_sizes"] + [hisLat_size]

        networks = []
        for layer_size in hidden_sizes:
            networks.append(nn.Linear(prev_size, layer_size))
            networks.append(nn.GELU())
            networks.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        networks.append(nn.Linear(prev_size, output_size))

        # Class Variables
        self.networks = nn.Sequential(*networks)

    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:    History input.

        Returns:
            ynn:    Output tensor.
        """

        # Flatten the input tensor
        znn = torch.flatten(xnn, start_dim=-2)

        # Forward pass
        if self.use_fpass == True:
            ynn = self.networks[:-1](znn)
        else:
            ynn = self.networks(znn)
        
        return ynn