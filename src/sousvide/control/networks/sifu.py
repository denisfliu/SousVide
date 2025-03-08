import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class SIFU(BaseNet):
    def __init__(self,
                 inputs: Dict[str,Dict[str,List[Union[str,int]]]],
                 input_size:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 feature_index:int,
                 dropout=0.1,
                 network_type="sifu"):
        """
        Initialize a Sequence Into Features (Unified) feedforward model.

        Args:
            inputs:         Inputs config.
            input_size:     Input size.
            hidden_sizes:   Hidden sizes.
            output_size:    Output size.
            feature_index:  Index of the feature to be extracted.
            dropout:        Dropout rate.
            network_type:   Type of network.

        Variables:
            network_type:   Type of network.
            input_indices:  Indices of the inputs.
            networks:       Network
            use_subnet:     Flag to use the subnet.
            subnet_idx:     Index of the subnet.
        """

        # Initialize the parent class
        super(SIFU, self).__init__()

        # Extract the inputs
        input_indices = nh.get_input_indices(inputs)
        
        # Check the arguments are valid
        assert feature_index < len(hidden_sizes), "Feature index out of range."
        assert input_size == nh.get_input_size(inputs["history"]), "Input size mismatch."
        
        # Populate the layers
        layers = []
        prev_size = input_size

        for idx,size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))

            if idx == feature_index:
                subnet_idx = len(layers)

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_size))

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.networks = nn.Sequential(*layers)
        self.use_subnet = True
        self.subnet_idx = subnet_idx

    def forward(self, xnn_hist:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hist:   History input.

        Returns:
            ynn:        Output tensor.
        """

        # Flatten the input tensor
        xnn = torch.flatten(xnn_hist, start_dim=-2)

        # Forward pass
        if self.use_subnet == True:
            ynn = self.networks[:self.subnet_idx](xnn)
        else:
            ynn = self.networks(xnn)
        
        return ynn