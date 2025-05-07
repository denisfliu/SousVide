import torch

from torch import nn

from sousvide.control.networks.base_net import BaseNet

class DRNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="drnet"):
        """
        Initialize a Direct Regression Network model.

        The network takes in a current input into an MLP
        and outputs a body rate command.

        """

        # Initialize the parent class
        super(DRNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        hidden_sizes = layers["hidden_sizes"]

        # Unpack network configs from parent
        in_sizes,fp_sizes,_ = self.get_io_sizes(expanded=True)

        # Populate the network
        prev_size = sum(in_sizes)
        output_size = fp_sizes[0]

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
    
    def forward(self,xnn_cr:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_cr: Current input.

        Returns:
            ynn:    Output tensor.
            ann:    Auxiliary outputs dictionary (if any).
        """
        
        # Command MLP
        ynn = self.networks["mlp"](xnn_cr)     

        return ynn,{}