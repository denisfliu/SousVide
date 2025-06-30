import torch

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class SIFU(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="sifu"):
        """
        Initialize a Sequence Into Features (Unified) feedforward model.

        The network takes history sequences and passes them through an
        MLP to estimate drone parameters (mass and thrust coefficient)
        during training. At runtime the network outputs the penultimate
        layer as a feature vector.

        """

        # Initialize the parent class
        super(SIFU, self).__init__(inputs,outputs,network_type)

        # Unpack some useful variables
        io_sizes = self.get_io_sizes()
        dropout = layers["dropout"]
        hidden_sizes = layers["hidden_sizes"]

        # Populate the network
        prev_size = io_sizes["xdp"]["dynamics"]
        mlp_sizes = hidden_sizes + [io_sizes["ydp"]["feature_vector"]]

        networks = []
        for layer_size in mlp_sizes:
            networks.append(nn.Linear(prev_size, layer_size))
            networks.append(nn.ReLU())
            networks.append(nn.Dropout(dropout))

            prev_size = layer_size
        
        networks.append(nn.Linear(prev_size, io_sizes["ypd"]["parameters"]))

        # Class Variables
        self.networks = nn.Sequential(*networks)

    def forward(self, Xnn:dict[str,torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            Xnn:    Input dictionary.

        Returns:
            Ynn:    Output dictionary.
        """

        # Unpack inputs
        xnn_dyn = Xnn["dynamics"]

        # Flatten the input tensor
        xnn = torch.flatten(xnn_dyn, start_dim=-2)

        # Forward pass
        ydp = self.networks[:-1](xnn)
        ypd = self.networks[-1](ydp)

        # Create the output dictionary
        Ynn = {"feature_vector": ydp,"parameters": ypd}

        return Ynn