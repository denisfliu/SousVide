import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union

class SIFS(nn.Module):
    def __init__(self,
                 inputs:Dict[str,Dict[str,List[Union[str,int]]]],
                 input_size:List[int],
                 shr_hidden_sizes:List[int],shr_output_size:int,mrg_hidden_sizes:List[int],
                 output_size:int,
                 feature_index:int,
                 dropout=0.1,
                 network_type="sifs"):
        """
        Initialize a Sequence Into Features (Shared) feedforward model.

        Args:
            inputs:             Inputs config.
            input_size:         Total input size.
            shr_hidden_sizes:   Hidden sizes of the shared network.
            shr_output_size:    Output size of the shared network.
            mrg_hidden_sizes:   Hidden sizes of the merged network.
            output_size:        Output size of the merged network.
            feature_index:      Index of the feature to be extracted.
            dropout:            Dropout rate.
            network_type:       Type of network.

        Variables:
            network_type:       Type of network.
            use_subnet:         Flag to use the subnet.
            subnet_idx:         Index of the subnet.
            shared_network:     Shared network.
            merged_network:     Merged network

        """

        # Initialize the parent class
        super(SIFS, self).__init__()

        # Extract the inputs
        input_indices = nh.get_input_indices(inputs)

        # Check the arguments are valid
        assert feature_index < len(mrg_hidden_sizes), "Feature index out of range."
        assert input_size[0]*input_size[1] == nh.get_input_size(inputs["history"]), "Input size mismatch."
        
        # Populate the shared layers
        shared_layers = []
        prev_size = len(inputs["history"][-1])

        for idx,size in enumerate(shr_hidden_sizes):
            shared_layers.append(nn.Linear(prev_size, size))

            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_size = size
        
        shared_layers.append(nn.Linear(prev_size, shr_output_size))

        # Populate the merged layers
        merged_layers = []
        prev_size = shr_output_size*len(inputs["history"][0])

        for idx,size in enumerate(mrg_hidden_sizes):
            merged_layers.append(nn.Linear(prev_size, size))

            if idx == feature_index:
                subnet_idx = len(merged_layers)

            merged_layers.append(nn.ReLU())
            merged_layers.append(nn.Dropout(dropout))
            prev_size = size

        merged_layers.append(nn.Linear(prev_size, output_size))

        # Combine the layers
        networks = nn.ModuleDict({
            "shared": nn.Sequential(*shared_layers),
            "merged": nn.Sequential(*merged_layers)
        })

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.networks = networks
        self.use_subnet = False
        self.subnet_idx = subnet_idx
    
    def forward(self, xnn_hist:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hist:   History input.

        Returns:
            ymn:        Output tensor.
        """

        # Forward pass on the shared network
        ysn = self.networks["shared"](xnn_hist)

        # Flatten the input tensor
        xmn = torch.flatten(ysn, start_dim=-2)

        # Forward pass on the merged network
        if self.use_subnet == True:
            ymn = self.networks["merged"][:self.subnet_idx](xmn)
        else:
            ymn = self.networks["merged"](xmn)

        return ymn