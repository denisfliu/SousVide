import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class SIFS(BaseNet):
    def __init__(self,
                 inputs:  Dict[str, List[List[Union[int, str]]]],
                 outputs: Dict[str, Dict[str, List[List[Union[int, str]]]]],
                 layers:  Dict[str, Union[int,List[int]]],
                 dropout=0.1,
                 network_type="sifs"):
        """
        Initialize a Sequence Into Features (Shared) feedforward model.

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
            
            use_fpass:      Use feature forward-pass.
            frame_len:      Frame length flag with size as value.
        """

        # Initialize the parent class
        super(SIFS, self).__init__()

        # Extract the inputs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs["fpass"])
        label_indices = nh.get_io_idxs(outputs["label"])
        
        shr_prev_size = len(input_indices["history"][-1])
        shr_hidden_sizes = layers["shr_hidden_sizes"][:-1]
        shr_output_size = layers["shr_hidden_sizes"][-1]
        
        mrg_prev_size = shr_output_size*len(input_indices["history"][0])
        mrg_hidden_sizes = layers["mrg_hidden_sizes"] + [layers["histLat_size"]]

        output_size = nh.get_io_size(label_indices)

        # Populate the shared networks
        shared_networks = []
        for layer_size in shr_hidden_sizes:
            shared_networks.append(nn.Linear(shr_prev_size, layer_size))
            shared_networks.append(nn.ReLU())
            shared_networks.append(nn.Dropout(dropout))

            shr_prev_size = layer_size
        
        shared_networks.append(nn.Linear(shr_prev_size, shr_output_size))

        # Populate the merged networks
        merged_networks = []
        for layer_size in mrg_hidden_sizes:
            merged_networks.append(nn.Linear(mrg_prev_size, layer_size))
            merged_networks.append(nn.ReLU())
            merged_networks.append(nn.Dropout(dropout))

            mrg_prev_size = layer_size
        
        merged_networks.append(nn.Linear(mrg_prev_size, output_size))

        # Combine the layers
        networks = nn.ModuleDict({
            "shared": nn.Sequential(*shared_networks),
            "merged": nn.Sequential(*merged_networks)
        })

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.networks = networks

        self.use_fpass = True
        self.nhy = max(input_indices["history"][0])+1
    
    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:    History input.

        Returns:
            ynn:    Output tensor.
        """

        # Forward pass on the shared network
        znn = self.networks["shared"](xnn)

        # Flatten the input tensor
        znn = torch.flatten(znn, start_dim=-2)

        # Forward pass on the merged network
        if self.use_fpass == True:
            ynn = self.networks["merged"][:-1](znn)
        else:
            ynn = self.networks["merged"](znn)

        return ynn