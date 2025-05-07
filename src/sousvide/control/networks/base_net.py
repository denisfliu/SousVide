import torch
import torch.nn as nn
import sousvide.control.network_helper as nh

from abc import ABC, abstractmethod

class BaseNet(nn.Module, ABC):
    """
    Base network module enforcing required attributes for child classes.

    Args:
        inputs:         Inputs config.
        outputs:        Outputs config.
        network_type:   Type of network.

    Variables:
        input_indices:  Indices of the input.
        fpass_indices:  Indices of the forward-pass output.
        label_indices:  Indices of the label output.
        network_type:   Type of network.
        Nx:             Input size.
        Ny:             Output size.
        nhy:            Frame length flag with size as value.
        use_fpass:      Use feature forward-pass.

        networks:       List of neural networks.
    """

    def __init__(self,
                 inputs:  dict[str, list[list[int|str]]],
                 outputs: dict[str, dict[str, list[list[int|str]]]],
                 network_type: str = "basenet"):
        super().__init__()

        # Some useful intermediate variables
        input_indices = nh.get_io_idxs(inputs)

        if "fpass" in outputs and "label" in outputs:
            fpass_indices = nh.get_io_idxs(outputs["fpass"])
            label_indices = nh.get_io_idxs(outputs["label"])
        elif "fpass" not in outputs and "label" not in outputs:
            fpass_indices = nh.get_io_idxs(outputs)
            label_indices = nh.get_io_idxs(outputs)
        else:
            raise ValueError("Both fpass and label outputs must be defined or neither.")

        # Check for history sequences and set nhy accordingly
        nhy: int = 0
        for input in inputs.values():
            seq_cand = input[0]
            if isinstance(seq_cand, list) and all(isinstance(x, int) for x in seq_cand):
                nhy = max(seq_cand[-1]+1, nhy)
        self.nhy = nhy

        # Define required attributes that all subclasses must implement
        self.input_indices = input_indices
        self.fpass_indices = fpass_indices
        self.label_indices = label_indices
        self.network_type = network_type
        self.nhy = nhy
        self.use_fpass = True
        
        # Initialize child specific attributes
        self.networks:nn.ModuleDict =  nn.ModuleDict()

    def get_io_dims(self) -> int:
        """
        Get the output size of the network.
        """

        Dim_x = nh.get_io_dims(self.input_indices)
        Dim_fp = nh.get_io_dims(self.fpass_indices)
        Dim_lb = nh.get_io_dims(self.label_indices)
        
        return Dim_x,Dim_fp,Dim_lb
    
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

        Nx = nh.get_io_size(self.input_indices,expanded)
        Ny_fp = nh.get_io_size(self.fpass_indices,expanded)
        Ny_lb = nh.get_io_size(self.label_indices,expanded)
        
        return Nx,Ny_fp,Ny_lb
    
    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        """
        Subclasses must implement a forward pass.
        """
        return None