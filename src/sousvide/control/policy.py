import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from typing import Dict,Tuple,Any
from sousvide.control.networks.base_net import BaseNet

class Policy(nn.Module):
    def __init__(self,
                 config:Dict[str,Dict[str,Any]],
                 name:str="policy"):
        
        # Initial Parent Call
        super(Policy,self).__init__()

        # Network Components
        networks:Dict[str,BaseNet] = nn.ModuleDict()
        nhy,Nz = 0,0
        for net_name,net_config in config["networks"].items():
            networks[net_name] = nf.generate_network(net_config)   

            network_inputs = net_config["inputs"]
            nhy = max(nhy,nh.get_max_length(network_inputs))
            if net_name == "featNet":
                Nz = net_config["output_size"]

        # Class Variables
        self.name = name
        self.networks = networks
        self.nhy = nhy
        self.Nz = Nz
        self.base_policy_inputs = ["rgb_image","objective","current","history","feature"]

    def forward(self, xnn_pol:Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            xnn_pol:    Policy variables.

        Returns:
            ynn:        Policy output.
            znn:        Feature output.
            xnn_pol:    (Updated) policy variables
        """

        # Forward Pass
        for name,network in self.networks.items():
            # Extract Inputs
            xnn_net = []
            for key,value in network.input_indices.items():
                xnn_net.append(nh.extract_inputs(xnn_pol[key],value))

            # Forward Pass
            ynn = network(*xnn_net)

            # Policy input update
            if name not in self.base_policy_inputs:
                xnn_pol[name] = ynn

        # Extract the feature vector if it exists
        if "featNet" in xnn_pol.keys():
            znn = xnn_pol["featNet"].squeeze()
        else:
            znn = None

        return ynn,znn,xnn_pol