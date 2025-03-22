import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class SVCN(BaseNet):
    def __init__(self,
                 inputs:  Dict[str, List[List[Union[int, str]]]],
                 outputs: Dict[str, List[List[Union[int, str]]]],
                 layers:  Dict[str, Union[int,List[int]]],
                 dropout=0.1,
                 network_type="svcn"):
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
        super(SVCN, self).__init__()

        # Extract the configs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs)
        label_indices = nh.get_io_idxs(outputs)

        Ncr =  len(input_indices["current"][-1])
        NfL,NhL = layers["featLat_size"],layers["histLat_size"]
        prev_size = Ncr + NfL + NhL 
        hidden_sizes = layers["hidden_sizes"]
        output_size = nh.get_io_size(label_indices)

        # Populate the network
        networks = []
        for layer_size in hidden_sizes:
            networks.append(nn.Linear(prev_size, layer_size))
            networks.append(nn.ReLU())
            networks.append(nn.Dropout(dropout))

            prev_size = layer_size

        networks.append(nn.Linear(prev_size, output_size))

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.networks = nn.Sequential(*networks)
    
    def forward(self,
                xnn_cr:torch.Tensor,
                xnn_fL:torch.Tensor,
                xnn_hL:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_cr: Current input.
            xnn_fL: Feature Network input.
            xnn_hL: History Network input.

        Returns:
            ynn:    Output tensor.
        """

        # Concatenate the inputs
        znn = torch.cat([xnn_cr,xnn_fL,xnn_hL],dim=1)
        
        # Feedforward
        ynn = self.networks(znn)         

        return ynn