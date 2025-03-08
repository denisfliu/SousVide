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
        self.network_type = None
        self.input_indices = None
        self.networks = None

    @abstractmethod
    def forward(self, x):
        """
        Subclasses must implement a forward pass.
        """
        pass