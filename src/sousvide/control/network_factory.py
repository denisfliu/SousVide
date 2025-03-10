from typing import Dict,Union,Any
from sousvide.control.networks.base_net import BaseNet
from sousvide.control.networks.mlp import MLP
from sousvide.control.networks.sifu import SIFU
from sousvide.control.networks.sifs import SIFS
from sousvide.control.networks.sift import SIFT
from sousvide.control.networks.sqfe import SqFE
from sousvide.control.networks.hpcn import HPCN
from sousvide.control.networks.svcn import SVCN

def generate_network(config:Dict[str,Any]) -> BaseNet:
    
    network_type = config["network_type"]

    if network_type == "mlp":
        network = MLP(**config)
    elif network_type == "sifu":
        network = SIFU(**config)
    elif network_type == "sifs":
        network = SIFS(**config)
    elif network_type == "sift":
        network = SIFT(**config)
    elif network_type == "sqfe":
        network = SqFE(**config)
    elif network_type == "svcn":
        network = SVCN(**config)
    elif network_type == "hpcn":
        network = HPCN(**config)
    elif network_type == None:
        network = None
    else:
        raise ValueError(f"Invalid network type: {config['network_type']}")
    
    return network
    