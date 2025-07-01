import torch
import numpy as np
from torch import nn
from torchvision.models import (
    squeezenet1_1
)
from sousvide.control.networks.base_net import BaseNet

class Pave(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="svnet"):
        """
        Initialize a SousVide Command Network.

        The network takes images, current states and histrory feature vector to
        output a command vector. The network uses a SqueezeNet backbone to extract
        image features.

        """

        # Initialize the parent class
        super(Pave, self).__init__(inputs,outputs,network_type)

        # Unpack network configs
        io_sizes = self.get_io_sizes()
        hidden_sizes = layers["hidden_sizes"]

        # Populate the actor network
        prev_size = io_sizes["xdp"]["dynamics"]

        actor = []
        for layer_size in hidden_sizes:
            actor.append(nn.Linear(prev_size, layer_size))
            actor.append(nn.ReLU())

            prev_size = layer_size
        actor.append(nn.Linear(prev_size,1))  # Output layer for command vector

        # Populate the critic network
        prev_size = io_sizes["xdp"]["dynamics"]

        critic = []
        for layer_size in hidden_sizes:
            critic.append(nn.Linear(prev_size, layer_size))
            critic.append(nn.ReLU())

            prev_size = layer_size
        critic.append(nn.Linear(prev_size, 1))

        # Populate the network
        networks = nn.ModuleDict({
            "actor": nn.Sequential(*actor),
            "critic": nn.Sequential(*critic),
        })

        # Class Variables
        self.networks = networks
        self.mode = "train"  # Default mode is training

    def forward(self,Xnn:dict[str,torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            Xnn:    Input dictionary.

        Returns:
            Ynn:    Output dictionary.
        """

        # Unpack inputs
        xnn = Xnn["dynamics"]
        unn_pr = xnn[0,11:15]
        xnn = xnn.view(xnn.size(0), -1)

        # Actor network
        ynn_res:torch.Tensor = self.networks["actor"](xnn).squeeze()

        # Add noise to the command vector during training
        if self.mode == "train":
            std_res = torch.Tensor([0.2, 0.05, 0.05, 0.05], device=ynn_res.device)
            dist = torch.distributions.Normal(ynn_res, std_res)
            ynn_res = dist.sample()
            l_prob = dist.log_prob(ynn_res)

            ynn_cm = unn_pr + ynn_res


        elif self.mode == "eval":
            ynn_cm = unn_pr + ynn_res

            l_prob = torch.zeros_like(ynn_cm)  # No noise, so log prob is zero

        # Compute the value of the current state
        ct_val = self.networks["critic"](Xnn["current"]).squeeze(-1)

        # Create the output dictionary
        Ynn = {
            "command": ynn_cm,
            "l_prob": l_prob,
            "ct_val": ct_val,
            }

        return Ynn