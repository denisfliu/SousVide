import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import List,Dict,Union

class BaseNet(nn.Module, ABC):
    """
    Base network module enforcing required attributes for child classes.

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
        networks:       List of neural networks.
        
        use_fpass:      Use feature forward-pass.
        Nznn:           Feature extractor flag with network (type,size) as value.
        nhy:            Frame length flag with size as value.
    """
    def __init__(self):
        super().__init__()

        # Define required attributes that all subclasses must implement
        self.network_type:str = "basenet"
        self.input_indices:Dict[str,List[torch.Tensor]] = {}
        self.fpass_indices:Dict[str,List[torch.Tensor]] = {}
        self.label_indices:Dict[str,List[torch.Tensor]] = {}
        self.networks:nn.ModuleDict =  nn.ModuleDict()
        
        self.use_fpass:bool = True
        self.Nznn:Union[int,Dict[str,List[int]]] = None
        self.nhy:int = 1

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Subclasses must implement a forward pass.
        """
        return None