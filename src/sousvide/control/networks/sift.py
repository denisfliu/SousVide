import torch
import sousvide.control.network_helper as nh

from torch import nn
from typing import List,Dict,Union
from sousvide.control.networks.base_net import BaseNet

class SIFT(BaseNet):
    def __init__(self,
                 inputs:  Dict[str, List[List[Union[int, str]]]],
                 outputs: Dict[str, Dict[str, List[List[Union[int, str]]]]],
                 layers:  Dict[str, Union[int,List[int]]],
                 dropout=0.1,
                 network_type="sift"):
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
        super(SIFT, self).__init__()

        # Extract the inputs
        input_indices = nh.get_io_idxs(inputs)
        fpass_indices = nh.get_io_idxs(outputs["fpass"])
        label_indices = nh.get_io_idxs(outputs["label"])

        d_model,d_ff = layers["d_model"],layers["d_ff"]
        num_heads,num_layers = layers["num_heads"],layers["num_layers"]
        output_size = nh.get_io_size(label_indices)

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
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.networks = networks
        self.position = self._generate_positional_encoding(d_model, len(inputs["history"][0]))
        
        self.use_fpass = True
        self.nhy = max(input_indices["history"][0])+1

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
    
    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:   History input.

        Returns:
            ynn:        Output tensor.
        """

        # Forward pass through embedding layer
        znn = self.networks["fc_in"](xnn) + self.position[:, :xnn.size(1), :].to(xnn.device)
        
        # Pass through the transformer
        znn = self.networks["encoder"](znn)

        if self.use_fpass == True:
            ynn = znn[:,-1,:]
        else:
            ynn = self.networks["fc_out"](znn[:,-1,:])
        
        return ynn