import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.mlp import MLP

class SqFE(BaseNet):
    def __init__(self,
                 inputs: Dict[str,Dict[str,List[Union[str,int]]]],
                 augment_layer:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 dropout=0.1,
                 Nsqn:int=1000,
                 network_type="sqfe"):
        """
        SqueezeNet Feature Extractor.

        Args:
            inputs:         Inputs config.
            augment_layer:  Layer where current state is injected.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            dropout:        Dropout rate.
            Nsqn:           Number of SqueezeNet outputs.
            network_type:   Type of network.

        Variables:
            network_type:   Type of network.
            inputs:         Inputs (overloaded).
            networks:       List of neural networks.
        """

        # Initialize the parent class
        super(SqFE, self).__init__()

        # Extract the inputs
        input_indices = nh.get_input_indices(inputs)
        Ncurr = nh.get_input_size(input_indices["current"])

        # Check the arguments are valid
        assert augment_layer < len(hidden_sizes), "Augment layer out of range."

        # Some useful intermediate variables
        networks = nn.ModuleDict({
            "cnn": squeezenet1_1()
        })

        if augment_layer == 0:
            networks["mlp0"] = nn.Identity()
            networks["mlp1"] = MLP(Nsqn+Ncurr,hidden_sizes,output_size,dropout)
        else:
            networks["mlp0"] = MLP(Nsqn,hidden_sizes[:augment_layer-1],hidden_sizes[augment_layer-1],dropout)
            networks["mlp1"] = MLP(hidden_sizes[augment_layer-1]+Ncurr,hidden_sizes[augment_layer:],output_size)
        
        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.networks = networks
        
    def forward(self, xnn_rgb:torch.Tensor, xnn_curr:torch.Tensor) ->  torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_rgb:    Image input.
            xnn_curr:   Current input.

        Returns:
            ynn:        Output tensor.
        """

        # Image CNN
        ycnn = self.networks["cnn"](xnn_rgb)

        # Feature Extractor MLPs
        yfe0 = self.networks["mlp0"](ycnn)
        ynn = self.networks["mlp1"](torch.cat((yfe0,xnn_curr),-1))

        return ynn