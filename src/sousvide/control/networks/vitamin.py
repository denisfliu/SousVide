import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class Vitamin(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 dropout=0.1,
                 network_type="vitamin"):
        """
        Initialize a Sequence Into Features (Transformer) model.
        
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
            position:       Positional encoding.
            
            use_fpass:      Use feature forward-pass.
            frame_len:      Frame length flag with size as value.
        """

        # Initialize the parent class
        super(Vitamin, self).__init__()

        # Extract the inputs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs)
        label_indices = nh.get_io_idxs(outputs)

        d_model,d_ff = layers["d_model"],layers["d_ff"]
        num_heads,num_layers = layers["num_heads"],layers["num_layers"]
        output_size = nh.get_io_size(label_indices)
        NhL = layers["histLat_size"]
        hidden_sizes = layers["hidden_sizes"]

        # Populate the layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu')

        fc_out = []
        prev_size = NhL+d_model
        for layer_size in hidden_sizes:
            fc_out.append(nn.Linear(prev_size, layer_size))
            fc_out.append(nn.ReLU())
            fc_out.append(nn.Dropout(dropout))

            prev_size = layer_size
        fc_out.append(nn.Linear(prev_size, output_size))

        networks = nn.ModuleDict({
            "fc_in": nn.Linear(len(inputs["history"][-1]), d_model),
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            "fc_out": nn.Sequential(*fc_out)
        })

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.networks = networks
        self.position = nh.generate_positional_encoding(d_model, len(inputs["history"][0])+1)
        
        self.use_fpass = True
        self.nhy = max(input_indices["history"][0])+1

    def forward(self,
                xnn_hy:torch.Tensor,
                xnn_cr:torch.Tensor,
                xnn_hL:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hy:       History input.
            xnn_cr:       Current input.
            xnn_hL:       History feature input.

        Returns:
            ynn:        Output tensor.
        """

        # Stack the inputs
        xnn = torch.cat((xnn_hy, xnn_cr.unsqueeze(1)), dim=1)

        # Forward pass through embedding layer
        znn = self.networks["fc_in"](xnn) + self.position[:, :xnn.size(1), :].to(xnn.device)
        
        # Pass through the transformer
        znn = self.networks["encoder"](znn)

        # Command mlp
        znn = torch.cat((znn[:,-1,:], xnn_hL), dim=1)
        ynn = self.networks["fc_out"](znn)
        
        return ynn