import torch
from torch import nn
from typing import List
from sousvide.control.networks.base_net import BaseNet

class MLP(BaseNet):
    def __init__(self,
                 input_size:int, hidden_sizes:List[int], output_size:int,
                 dropout=0.1,network_type="mlp"):
        """
        Initialize a simple MLP model.

        Args:
            input_size:     Input size.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            dropout:        Dropout rate.

        Variables:
            network_type:   Type of network.
            networks:       List of neural networks.
        """
        # Initialize the parent class
        super(MLP, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        # Define the model
        self.network_type = network_type
        self.input_indices = {}
        self.networks = nn.Sequential(*layers)
    
    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Output tensor.
        """

        # Simple MLP
        ynn = self.networks(xnn)         

        return ynn