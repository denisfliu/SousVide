import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from typing import Dict,Tuple,Any,Union,List
from sousvide.control.networks.base_net import BaseNet

class Policy(nn.Module):
    def __init__(self,
                 config:Dict[str,Dict[str,Any]],
                 name:str="policy"):
        
        # Initial Parent Call
        super(Policy,self).__init__()

        # Populate the network
        networks:Dict[str,BaseNet] = nn.ModuleDict()
        nhy,Nz = 0,0
        for net_name,net_config in config["networks"].items():
            networks[net_name] = nf.generate_network(net_config)   

            # Update the max sequence length variable
            network_inputs = net_config["inputs"]
            nhy = max(nhy,nh.get_max_length(network_inputs))

            # Additional configuration for featNets
            if net_name == "featNet":
                Nz = net_config["layers"]["featLat_size"]
            
        # Class Variables
        self.name = name
        self.networks = networks
        self.nhy,self.Nz = nhy,Nz

    def forward(self, xnn_pol:Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            xnn_pol:    Policy variables.

        Returns:
            ynn:        Policy output.
            znn:        Feature output.
            io_nn:      Network inputs/outputs.
        """

        # Initialize the policy outputs
        ynn,znn,io_nn = None,None,{}

        # Forward Pass through the networks
        for name,network in self.networks.items():
            # Extract the Network Inputs
            xnn_net = []
            for key,value in network.input_indices.items():
                xnn_net.append(nh.extract_io(xnn_pol[key],value))

            # Forward Pass through the Network
            ynn_net = network(*xnn_net)

            # Store the inference inputs/outputs
            io_nn[name] = {
                "inputs" : xnn_net,
                "outputs": network.label_indices
            }

            # Update xnn_pol if network feeds into another network
            for fpass_key in network.fpass_indices.keys():
                if fpass_key not in xnn_pol.keys():
                    xnn_pol[fpass_key] = ynn_net

        # Extract the policy tensor outputs
        ynn = xnn_pol["command"]

        if "featLat" in xnn_pol.keys():
            znn = xnn_pol["featLat"]

        return ynn,znn,io_nn