import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class SIFT(BaseNet):
    def __init__(self,
                 inputs: Dict[str,Dict[str,List[Union[str,int]]]],
                 d_model:int, num_heads:int, d_ff:int, num_layers:int,
                 output_size:int,
                 dropout=0.1,
                 network_type="sift"):
        """
        Initialize a Sequence Into Features (Transformer) model.
        
        Args:
            inputs:         Inputs config.
            num_heads:      Number of heads.
            d_ff:           Dimension of the feedforward layer.
            num_layers:     Number of layers.
            output_size:    Output size.
            dropout:        Dropout rate.
            network_type:   Type of network.

        Variables:
            position:       Positional encoding.
            fc_in:          Input layer.
            encoder:        Encoder.
            fc_out:         Output layer

        """

        # Initialize the parent class
        super(SIFT, self).__init__()

        # Extract the inputs
        input_indices = nh.get_input_indices(inputs)

        # Populate the layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu')
        
        networks = nn.ModuleDict({
            "fc_in": nn.Linear(len(inputs["history"][-1]), d_model),
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            "fc_out": nn.Linear(d_model, output_size)
        })

        # Define the model
        self.network_type = network_type
        self.input_indices = input_indices
        self.networks = networks
        self.position = self._generate_positional_encoding(d_model, len(inputs["history"][0]))

    def _generate_positional_encoding(self, d_model, max_seq_len):
        """
        Generate positional encoding.

        Args:
            d_model:        Dimension of the model.
            max_seq_len:    Maximum sequence length.

        Returns:
            pe:             Positional encoding.

        """

        # Generate the positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
    
    def forward(self, xnn_hist:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hist:   History input.

        Returns:
            ynn:        Output tensor.
        """

        # Forward pass through embedding layer
        zem = self.networks["fc_in"](xnn_hist) + self.position[:, :xnn_hist.size(1), :].to(xnn_hist.device)
        
        # Pass through the transformer
        ztf = self.networks["encoder"](zem)

        # Transform embedding into output tensor
        ynn = self.networks["fc_out"](ztf[:,-1,:])
        
        return ynn