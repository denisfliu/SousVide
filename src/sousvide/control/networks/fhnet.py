import torch

from torch import nn

from sousvide.control.networks.base_net import BaseNet

class FHNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="drnet"):
        """
        Initialize a Flat Horizon Network model.

        The network takes in a current input into an MLP
        and outputs a body rate command.

        """

        # Initialize the parent class
        super(FHNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        hzn_hidden_sizes = layers["hzn_hidden_sizes"]
        cmd_hidden_sizes = layers["cmd_hidden_sizes"]

        # Unpack network configs from parent
        in_sizes,fp_sizes,lb_sizes = self.get_io_sizes(expanded=True)

        # Populate the network
        prev_size = sum(in_sizes)
        output_size = lb_sizes[0]

        mlp_hzn = []
        for layer_size in hzn_hidden_sizes:
            mlp_hzn.append(nn.Linear(prev_size, layer_size))
            mlp_hzn.append(nn.ReLU())
            mlp_hzn.append(nn.Dropout(dropout))

            prev_size = layer_size
        mlp_hzn.append(nn.Linear(prev_size, output_size))

        prev_size = output_size
        output_size = lb_sizes[1]

        mlp_cmd = []
        for layer_size in cmd_hidden_sizes:
            mlp_cmd.append(nn.Linear(prev_size, layer_size))
            mlp_cmd.append(nn.ReLU())
            mlp_cmd.append(nn.Dropout(dropout))

            prev_size = layer_size
        mlp_cmd.append(nn.Linear(prev_size, output_size))

        # Populate the network
        networks = nn.ModuleDict({
            "mlp_hzn": nn.Sequential(*mlp_hzn),
            "mlp_cmd": nn.Sequential(*mlp_cmd),
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
        znn = self.networks["mlp_hzn"](xnn_cr)
        ynn = self.networks["mlp_cmd"](znn)

        if self.use_fpass == False:
            ynn = torch.concat([znn, ynn], dim=-1)

        return ynn,{}