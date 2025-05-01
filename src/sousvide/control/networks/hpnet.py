import torch
import sousvide.control.network_helper as nh

from torch import nn
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet

class HPNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="hpnet"):
        """
        Initialize a HotPot Network model.

        The network takes in a current input and a history inputs into an MLP
        and outputs a motor command.

        """

        # Initialize the parent class
        super(HPNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        hidden_sizes = layers["hidden_sizes"]

        # Unpack network configs from parent
        input_sizes,_,output_sizes = self.get_io_sizes(expanded=True)
        input_size = sum(input_sizes)
        output_size = output_sizes[0]

        # Populate the network
        prev_size = input_size
        mlp = []
        for layer_size in hidden_sizes:
            mlp.append(nn.Linear(prev_size, layer_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))

            prev_size = layer_size
        mlp.append(nn.Linear(prev_size, output_size))

        # Populate the network
        networks = nn.ModuleDict({
            "mlp": nn.Sequential(*mlp),
        })

        # Class Variables
        self.networks = networks
    
    def forward(self,
                xnn_cr:torch.Tensor,
                xnn_hL:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_cr: Current input (second instance).
            xnn_hL: History Network input.

        Returns:
            ynn:    Output tensor.
        """
        
        # Command MLP
        xnn = torch.cat((xnn_cr, xnn_hL), dim=1)
        ynn = self.networks["mlp"](xnn)     

        return ynn