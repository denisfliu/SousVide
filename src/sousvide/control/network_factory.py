import os
import json
import torch
import sousvide.control.network_helper as nh

from typing import Dict,Union,List
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.mlp import MLP
from sousvide.control.networks.sifu import SIFU
from sousvide.control.networks.sifs import SIFS
from sousvide.control.networks.sift import SIFT
from sousvide.control.networks.sqfe import SqFE
from sousvide.control.networks.hpcn import HPCN
from sousvide.control.networks.svcn import SVCN

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
        Nz:         The number of features (if any).
    """

    # Some useful intermediate variables
    network_type = net_config["network_type"]
    network_path = os.path.join(pilot_path,net_name+".pt")

    # If the network already exists, load it. Otherwise, create it.
    if os.path.isfile(network_path):
        network = torch.load(network_path)
    else:
        # Simple Networks
        if network_type == "mlp":
            network = MLP(**net_config)
        # Feature Extractors
        elif network_type == "sqfe":
            network = SqFE(**net_config)
        # History Networks
        elif network_type == "sifu":
            network = SIFU(**net_config)
        elif network_type == "sifs":
            network = SIFS(**net_config)
        elif network_type == "sift":
            network = SIFT(**net_config)
        # Command Networks
        elif network_type == "svcn":
            network = SVCN(**net_config)
        elif network_type == "hpcn":
            network = HPCN(**net_config)
        else:
            raise ValueError(f"Invalid network type: {net_config['network_type']}")
    
    # Check if network has feature,sequence requirments
    nhy,Nz = network.nhy,network.Nznn
    
    return network,nhy,Nz
    