import torch
import torch.nn as nn
import sousvide.control.network_helper as nh

from abc import ABC, abstractmethod

class BaseNet(nn.Module, ABC):
    """
    Base network module enforcing required attributes for child classes.

    Args:
        inputs:         Inputs config.
        prediction:     Prediction config.
        deployment:     Deployment config.
        network_type:   Type of network.

    Variables:
        in_idxs:        Indices of the input.
        pd_idxs:        Indices of the forward-pass output.
        dp_idxs:        Indices of the label output.
        network_type:   Type of network.
        Nhy:            Maximum history length.
        use_fpass:      Use feature forward-pass.

        networks:       List of neural networks.
    """

    def __init__(self,
                 inputs: dict[str, dict[str,list[int|list[int|str]]]],
                 outputs: dict[str, dict[str,list[int|list[int|str]]]],
                 network_type: str ):
        super().__init__()

        # Some useful intermediate variables
        xpd_idxs = nh.get_io_idxs(inputs)
        ypd_idxs

        pd_idxs = nh.get_io_idxs(prediction)
        dp_idxs = nh.get_io_idxs(deployment)

        # Check for history sequences and set Nhy accordingly
        Nhy: int = 0
        for input in inputs.values():
            seq_cand = input[0]
            if isinstance(seq_cand, list) and all(isinstance(x, int) for x in seq_cand):
                Nhy_cand = seq_cand[-1]+1
                Nhy = max(Nhy_cand, Nhy)

        # Define required attributes that all subclasses must implement
        self.in_idxs = in_idxs
        self.pd_idxs = pd_idxs
        self.dp_idxs = dp_idxs
        self.network_type = network_type
        self.Nhy = Nhy
        self.use_fpass = True
        
        # Initialize child specific attributes
        self.networks:nn.ModuleDict =  nn.ModuleDict()

    def get_io_dims(self) -> int:
        """
        Get the output size of the network.
        """

        in_dims = nh.get_io_dims(self.in_idxs)
        pd_dims = nh.get_io_dims(self.pd_idxs)
        dp_dims = nh.get_io_dims(self.dp_idxs)
        
        return in_dims,pd_dims,dp_dims
    
    def get_io_sizes(self,expanded:bool=False) -> int:
        """
        Get the output size of the network.

        Args:
            expanded:   If True, return the expanded size of the network.
                        If False, return the size of the network.

        Returns:
            Nx:         Input size.
            Ny_fp:      Forward-pass output size.
            Ny_lb:      Label output size.
        """

        in_sizes = nh.get_io_size(self.in_idxs,expanded)
        pd_sizes = nh.get_io_size(self.pd_idxs,expanded)
        dp_sizes = nh.get_io_size(self.dp_idxs,expanded)
        
        return in_sizes,pd_sizes,dp_sizes
    
    @abstractmethod
    def forward(self, x) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
        """
        Subclasses must implement a forward pass.
        """
        return None,None