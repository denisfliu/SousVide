import os
import json
import torch
import sousvide.control.network_helper as nh

from typing import Dict,Union,List
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.sifu import SIFU
from sousvide.control.networks.sift import SIFT
from sousvide.control.networks.hpnet import HPNet
from sousvide.control.networks.svnet import SVNet
from sousvide.control.networks.afnet import AFNet
from sousvide.control.networks.jrnetv1 import JRNetv1
from sousvide.control.networks.jrnetv2 import JRNetv2
from sousvide.control.networks.jrnetv3 import JRNetv3
from sousvide.control.networks.jrnetv4 import JRNetv4
from sousvide.control.networks.qznet import QZNet

def generate_network(
        net_config:Dict[str,Union[str,Dict[str,List[List[Union[str,int]]]]]],
        net_name:str,
        pilot_path:str) -> BaseNet:
    """
    Generate a network based on the configuration dictionary. If the network does
    not already exist as a .pth file, it will be created and saved to the specified
    path. Only one network of each type can exist per pilot.

    Args:
        config:     Configuration dictionary for the network.
        net_name:   Name of the network.
        pilot_path: Pilot path.

    Returns:
        network:    The generated network.
        nhy:        The maximum sequence length (if any).
    """

    # Some useful intermediate variables
    network_type = net_config["network_type"]
    network_path = os.path.join(pilot_path,net_name+".pt")

    # If the network already exists, load it. Otherwise, create it.
    if os.path.isfile(network_path):
        network = torch.load(network_path)
    else:
        # Simple Networks
        if network_type == "simple":
            network = BaseNet(**net_config)
        # Feature Extractors
        # ==========>
        # History Networks
        elif network_type == "sifu":
            network = SIFU(**net_config)
        elif network_type == "sift":
            network = SIFT(**net_config)
        # Command Networks
        elif network_type == "hpnet":
            network = HPNet(**net_config)
        elif network_type == "svnet":
            network = SVNet(**net_config)
        elif network_type == "afnet":
            network = AFNet(**net_config)
        elif network_type == "jrnetv1":
            network = JRNetv1(**net_config)
        elif network_type == "jrnetv2":
            network = JRNetv2(**net_config)
        elif network_type == "jrnetv3":
            network = JRNetv3(**net_config)
        elif network_type == "jrnetv4":
            network = JRNetv4(**net_config)
        elif network_type == "qznet":
            network = QZNet(**net_config)
        # Invalid Network Type
        else:
            raise ValueError(f"Invalid network type: {net_config['network_type']}")
    
    # Check if network has feature,sequence requirments
    nhy = network.nhy
    
    return network,nhy
    