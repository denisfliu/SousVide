import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class HPCN(BaseNet):
    def __init__(self,
                 inputs: Dict[str,Dict[str,List[Union[str,int]]]],
                 input_size:int, hidden_sizes:List[int], output_size:int,
                 dropout=0.1,network_type="mlp"):
        """
        Initialize a HotPot Command Network.

        Args:
            inputs:         Inputs config.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            dropout:        Dropout rate.

        Variables:
            network_type:   Type of network.
            inputs:         Inputs (overloaded).
            networks:       List of neural networks.
        """

        # Initialize the parent class
        super(HPCN, self).__init__()

        # Extract the inputs
        input_indices = nh.get_input_indices(inputs)
        
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
        self.input_indices = input_indices
        self.networks = nn.Sequential(*layers)
    
    def forward(self,
                xnn_obj:torch.Tensor,
                xnn_curr:torch.Tensor,
                xnn_hNet:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_obj:    Objective input.
            xnn_curr:   Current input.
            xnn_fNet:   Feature Network input.
            xnn_hNet:   History Network input.

        Returns:
            ynn:        Output tensor.
        """

        # Concatenate the inputs
        xnn = torch.cat([xnn_obj,xnn_curr,xnn_hNet],dim=1)
        
        # Feedforward
        ynn = self.networks(xnn)         

        return ynn