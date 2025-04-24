import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class SIFU(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="sifu"):
        """
        Initialize a Sequence Into Features (Unified) feedforward model.

        The network takes history sequences and passes them through an
        MLP to estimate drone parameters (mass and thrust coefficient)
        during training. At runtime the network outputs the penultimate
        layer as a feature vector.

        """

        # Initialize the parent class
        super(SIFU, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]

        # Unpack network configs from parent
        prev_size,hisLat_size,output_size = self.get_io_sizes()

        # Populate the network
        hidden_sizes = layers["hidden_sizes"] + [hisLat_size]

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