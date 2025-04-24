import torch
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class JRNetv1(BaseNet):
    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 layers:  dict[str, int|list[int]],
                 network_type="jrnetv1"):
        """
        Initialize a Jorim v1 Network model.

        The network takes in a current input and a history input into
        a transformer and outputs a motor command.
        """

        # Initialize the parent class
        super(JRNetv1, self).__init__(inputs,outputs,network_type)

        # Unpack network configs from config
        dropout = layers["dropout"]
        d_model,d_ff = layers["d_model"],layers["d_ff"]
        num_heads,num_layers = layers["num_heads"],layers["num_layers"]

        # Unpack network configs from parent
        input_dims,_,_ = self.get_io_dims()
        _,_,output_sizes = self.get_io_sizes(expanded=True)
        d_state,d_input = input_dims[0][1], input_dims[1][1]
        output_size = output_sizes[0]

        # Populate the network
        state_embed = nn.Linear(d_state, d_model)
        input_embed = nn.Linear(d_input, d_model)
        pos_encoding = nh.generate_positional_encoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu')
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size),
        )

        # Class Variables
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        self.networks = nn.ModuleDict({
            "state_embed": state_embed,
            "input_embed": input_embed,
            "decoder": decoder,
            "head": head
        })


    def forward(self,
                xnn_hy1:torch.Tensor,
                xnn_hy2:torch.Tensor,
                xnn_cr:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn_hy1:    History input.
            xnn_hy2:    History input.
            xnn_cr:     Current input.

        Returns:
            ynn:        Output tensor.
        """

        # Package the inputs
        xnn_st = torch.cat((xnn_cr.unsqueeze(1), xnn_hy1), dim=1)
        xnn_in = xnn_hy2
        Nwd_st,Nwd_in = xnn_st.shape[1],xnn_in.shape[1]
        
        tgt = self.networks["state_embed"](xnn_st) + self.pos_encoding[:,:Nwd_st,:]
        mem = self.networks["input_embed"](xnn_in) + self.pos_encoding[:,:Nwd_in,:]

        tgt_mask = torch.triu(torch.full((Nwd_st, Nwd_st), float('-inf')), diagonal=1)
        tgt_mask = tgt_mask.to(xnn_st.device)

        znn = self.networks["decoder"](tgt, mem, tgt_mask=tgt_mask)

        ynn = self.networks["head"](znn[:,-1,:])
        
        return ynn