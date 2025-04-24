import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class SIFT(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="sift"):
        """
        Initialize a Sequence Into Features (Transformer) model.
        
        The network takes history sequences and passes them through an
        MLP to estimate drone parameters (mass and thrust coefficient)
        during training. At runtime the network outputs the penultimate
        layer as a feature vector.
        
        """
        
        # Initialize the parent class
        super(SIFT, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        d_model,d_ff = layers["d_model"],layers["d_ff"]
        num_heads,num_layers = layers["num_heads"],layers["num_layers"]
        prev_size = layers["d_model"]

        # Unpack network configs from parent
        input_dims,_,_ = self.get_io_dims()
        _,fp_sizes,lb_sizes = self.get_io_sizes(expanded=True)
        input_size = input_dims[0][1]
        hisLat_size,output_size = fp_sizes[0],lb_sizes[0]

        # Populate the network
        hidden_sizes = layers["hidden_sizes"] + [hisLat_size]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu')
        
        mlp_out = []
        for layer_size in hidden_sizes:
            mlp_out.append(nn.Linear(prev_size, layer_size))
            mlp_out.append(nn.ReLU())
            mlp_out.append(nn.Dropout(dropout))

            prev_size = layer_size
        mlp_out.append(nn.Linear(prev_size, output_size))
        
        networks = nn.ModuleDict({
            "fc_in": nn.Linear(input_size, d_model),
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            "fc_out": nn.Sequential(*mlp_out)
        })

        # Class Variables
        self.networks = networks
        self.position = nh.generate_positional_encoding(d_model, len(inputs["history"][0]))

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
        znn = self.networks["encoder"](znn)[:,-1,:]

        if self.use_fpass == True:
            ynn = self.networks["fc_out"][:-1](znn)
        else:
            ynn = self.networks["fc_out"](znn)
        
        return ynn