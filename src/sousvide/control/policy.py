import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from typing import Dict,Tuple,Any
from sousvide.control.networks.base_net import BaseNet

class Policy(nn.Module):
    def __init__(self,
                 config:Dict[str,Dict[str,Any]],
                 device:torch.device,
                 name:str="policy"):
        
        # Initial Parent Call
        super(Policy,self).__init__()

        # Class Variables
        self.name = name
        self.Nz = 0
        self.policy_inputs = ["rgb_image","objective","current","history","feature"]

        # Network Components
        self.networks:Dict[str,BaseNet] = nn.ModuleDict()
        for key in config["networks"].keys():
            self.networks[key] = nf.generate_network(config["networks"][key])   
            
        # Send to Device
        self.to(device)
        
    def forward(self, xnn_pol:Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
        """
        Forward pass of the model.
        """

        # Forward Pass
        for net_name,network in self.networks.items():
            # Extract Inputs
            xnn_net = []
            for key,value in network.input_indices.items():
                xnn_net.append(nh.extract_inputs(xnn_pol[key],value))

            # Forward Pass
            ynn = network(*xnn_net)

            # Policy input update
            if net_name not in self.policy_inputs:
                xnn_pol[net_name] = ynn

        return ynn,xnn_pol