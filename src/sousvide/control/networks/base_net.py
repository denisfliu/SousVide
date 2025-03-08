import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict,Tuple,Any,List

class BaseNet(nn.Module, ABC):
    """
    Base network module enforcing required attributes for child classes.
    """
    def __init__(self):
        super().__init__()

        # Define required attributes that all subclasses must implement
        self.network_type = "basenet"
        self.input_indices = {}
        self.networks =  nn.ModuleDict()

    @abstractmethod
    def forward(self, x):
        """
        Subclasses must implement a forward pass.
        """
        pass