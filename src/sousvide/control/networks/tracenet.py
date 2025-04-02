import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.mlp import MLP

from sousvide.control.networks.base_net import BaseNet

class TraceNet(BaseNet):
    def __init__(self,
                 inputs:  Dict[str, List[List[Union[int, str]]]],
                 outputs: Dict[str, List[List[Union[int, str]]]],
                 layers:  Dict[str, Union[int,List[int]]],
                 dropout=0.1,Nsq=1000,
                 network_type="tracenet"):
        """
        Initialize a SousVide TraceNet.

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
        super(TraceNet, self).__init__()

        # Extract the configs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs)
        label_indices = nh.get_io_idxs(outputs)

        Ncr =  len(input_indices["current"][-1])
        hidden_sizes = layers["hidden_sizes"]
        augment_layer = layers["augment_layer"]
        output_size = nh.get_io_size(label_indices)

        # Populate the network
        networks = nn.ModuleDict({
            "cnn": squeezenet1_1()
        })

        if augment_layer == 0:
            networks["mlp0"] = nn.Identity()
            networks["mlp1"] = MLP(Nsq+Ncr,hidden_sizes,output_size,dropout)
        else:
            networks["mlp0"] = MLP(Nsq,hidden_sizes[:augment_layer-1],hidden_sizes[augment_layer-1],dropout)
            networks["mlp1"] = MLP(hidden_sizes[augment_layer-1]+Ncr,hidden_sizes[augment_layer:],output_size)
        
        # Define the model        
        self.network_type = network_type
        self.input_indices = input_indices
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.networks = networks
        
        # self.Nznn = output_size
    
    def forward(self,
                xnn_im:torch.Tensor,
                xnn_cr:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_im:    Image input.
            xnn_cr:    Current input.

        Returns:
            ynn:    Output tensor.
        """

        # Image CNN
        znn = self.networks["cnn"](xnn_im)

        # Feature Extractor MLPs
        znn = self.networks["mlp0"](znn)
        znn = torch.cat((znn,xnn_cr),-1)
        ynn = self.networks["mlp1"](znn)

        return ynn