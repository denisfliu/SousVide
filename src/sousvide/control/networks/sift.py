import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List

class SIFT(nn.Module):
    def __init__(self,
                 history_inputs:List[str],history_frames:int,
                 d_model:int, num_heads:int, d_ff:int, num_layers:int,
                 output_size:int,
                 dropout=0.1,
                 network_type="sift"):
        """
        Initialize a Sequence Into Features (Transformer) model.
        
        Args:
            history_inputs: List of inputs.
            history_frames: Number of frames.
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

        # Populate the layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu')
        
        # Define the model
        self.network_type = network_type
        self.inputs = {}

        self.position = self._generate_positional_encoding(d_model, len(history_frames))
        self.input_idx = nh.get_indices(history_inputs,history_frames)

        self.networks = nn.ModuleDict({
            "fc_in": nn.Linear(len(history_inputs), d_model),
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            "fc_out": nn.Linear(d_model, output_size)
        })
        
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
    
    def forward(self, history:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            history:  Input tensor.

        Returns:
            ynn:  Output tensor.
        """

        # Extract the inputs
        xnn = nh.extract_array_inputs(history,self.input_idx)

        # Transform input tensor into embedding
        znn = self.networks["fc_in"](xnn) + self.position[:, :xnn.size(1), :].to(xnn.device)
        
        # Pass through the transformer
        znn = self.networks["encoder"](znn)

        # Transform embedding into output tensor
        ynn = self.networks["fc_out"](znn[:,-1,:])
        
        return ynn