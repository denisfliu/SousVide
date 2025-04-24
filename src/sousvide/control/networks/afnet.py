import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet
import sousvide.visualize.rich_utilities as ru
console = ru.get_console()

class AFNet(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="vitamin"):
        """
        Initialize a Air Fryer Network model.

        The network takes in a current input and a history input into
        a transformer and outputs a motor command.
        """

        # Initialize the parent class
        super(AFNet, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        d_model,d_ff = layers["d_model"],layers["d_ff"]
        num_heads,num_layers = layers["num_heads"],layers["num_layers"]

        # Unpack network configs from parent
        input_dims,_,_ = self.get_io_dims()
        _,_,output_sizes = self.get_io_sizes(expanded=True)
        d_input = input_dims[1][0]
        output_size = output_sizes[0]

        # Populate the network
        embedding = nn.Linear(d_input, d_model)
        pos_encoding = nh.generate_positional_encoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu')
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size),
        )

        # Class Variables
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        self.networks = nn.ModuleDict({
            "embedding": embedding,
            "encoder": encoder,
            "head": head
        })


    def forward(self,
                xnn_hy:torch.Tensor,
                xnn_cr:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hy:       History input.
            xnn_cr:       Current input.

        Returns:
            ynn:        Output tensor.
        """

        
        # Stack the inputs
        xnn = torch.cat((xnn_cr.unsqueeze(1),xnn_hy), dim=1)
        Nwd = xnn.shape[1]

        # Forward pass through embedding layer
        znn = self.networks["embedding"](xnn) + self.pos_encoding[:, :Nwd, :]
        
        # Pass through the transformer
        znn = self.networks["encoder"](znn)

        # Command mlp
        ynn = self.networks["head"](znn[:,-1,:])
        
        return ynn