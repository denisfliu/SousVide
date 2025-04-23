import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class SIFU(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 dropout=0.1,
                 network_type="sifu"):
        """
        Initialize a Sequence Into Features (Unified) feedforward model.

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
        super(SIFU, self).__init__()

        # Extract the configs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs["fpass"])
        label_indices = nh.get_io_idxs(outputs["label"])

        prev_size = nh.get_io_size(input_indices)
        hidden_sizes = layers["hidden_sizes"] + [layers["histLat_size"]]
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
        
        self.use_fpass = True
        self.nhy = len(inputs["history"][0])

    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:    History input.

        Returns:
            ynn:    Output tensor.
        """

        # Flatten the input tensor
        znn = torch.flatten(xnn, start_dim=-2)

        # Forward pass
        if self.use_fpass == True:
            ynn = self.networks[:-1](znn)
        else:
            ynn = self.networks(znn)
        
        return ynn