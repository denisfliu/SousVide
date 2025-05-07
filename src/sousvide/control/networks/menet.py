import torch

from torch import nn

from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.ensembles import (
    Expert,
    Gating
    )

class MENet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="menet"):
        """
        Initialize a Mixture of Experts Network model.

        The network takes in a current input into an MLP
        and outputs a body rate command.

        """

        # Initialize the parent class
        super(MENet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        exp_hidden_sizes = layers["exp_hidden_sizes"]
        cmd_hidden_sizes = layers["cmd_hidden_sizes"]
        num_experts = layers["number_of_experts"]

        # Unpack network configs from parent
        in_sizes,fp_sizes,lb_sizes = self.get_io_sizes(expanded=True)

        # Populate the network
        exp_in_dim,exp_out_dim = in_sizes[0],lb_sizes[0]
        cmd_in_dim,cmd_out_dim = exp_out_dim,fp_sizes[0]

        # Populate the network
        networks = nn.ModuleDict({
            "gating": Gating(exp_in_dim, num_experts),
            "experts": nn.ModuleList([Expert(exp_in_dim, exp_hidden_sizes, exp_out_dim) for _ in range(num_experts)]),
            "mlp_cmd": Expert(cmd_in_dim, cmd_hidden_sizes, cmd_out_dim),
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
        
        # Forward pass
        wnn = self.networks["gating"](xnn_cr)
        enns = torch.stack([e(xnn_cr) for e in self.networks["experts"]], dim=1)
        znn = torch.sum(wnn.unsqueeze(-1) * enns, dim=1)
        ynn = self.networks["mlp_cmd"](znn)

        # Pack Outputs
        if self.use_fpass == False:
            ynn = torch.concat([znn, ynn], dim=-1)
        ann = {"weights": wnn, "experts": enns}
        
        return ynn,ann