import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import List,Dict,Union

class BaseNet(nn.Module, ABC):
    """
    Base network module enforcing required attributes for child classes.
    """
    def __init__(self):
        super().__init__()

        # Define required attributes that all subclasses must implement
        self.network_type:str = "basenet"
        self.input_indices:Dict[str,List[torch.Tensor]] = {}
        self.fpass_indices:Dict[str,List[torch.Tensor]] = {}
        self.label_indices:Dict[str,List[torch.Tensor]] = {}
        self.networks:nn.ModuleDict =  nn.ModuleDict()

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Subclasses must implement a forward pass.
        """
        return None