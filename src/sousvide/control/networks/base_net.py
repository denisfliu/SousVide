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
        # Initial Parent Call
        super().__init__()

        # Check if prediction and deployment are provided
        if not ("prediction" in inputs and "deployment" in inputs):
            inputs = {"prediction": inputs,"deployment": inputs}

        if not ("prediction" in outputs and "deployment" in outputs):
            outputs = {"prediction": outputs,"deployment": outputs}

        # Some useful intermediate variables
        xpd_idxs,ypd_idxs = nh.get_io_idxs(inputs["prediction"]),nh.get_io_idxs(outputs["prediction"])
        xdp_idxs,ydp_idxs = nh.get_io_idxs(inputs["deployment"]),nh.get_io_idxs(outputs["deployment"])
        
        # Check for history sequences and set Nhy accordingly
        Nhy: int = 0
        for input in inputs.values():
            for input_item in input.values():
                seq_cand = input_item[0]
                if isinstance(seq_cand, list) and all(isinstance(x, int) for x in seq_cand):
                    Nhy_cand = seq_cand[-1]+1
                    Nhy = max(Nhy_cand, Nhy)

        # Define required attributes that all subclasses must implement
        self.network_type = network_type
        self.io_idxs = {
            "xpd": xpd_idxs, "ypd": ypd_idxs,
            "xdp": xdp_idxs, "ydp": ydp_idxs
        }
        self.Nhy = Nhy
        self.use_deploy = True

        # Initialize child specific attributes
        self.networks:nn.ModuleDict =  nn.ModuleDict()

    def get_io_dims(self) -> int:
        """
        Get the output size of the network.

        Returns:
            io_dims:    Dictionary of input/output dimensions.
        """

        xpd_dims = nh.get_io_dims(self.io_idxs["xpd"])
        ypd_dims = nh.get_io_dims(self.io_idxs["ypd"])
        xdp_dims = nh.get_io_dims(self.io_idxs["xdp"])
        ydp_dims = nh.get_io_dims(self.io_idxs["ydp"])

        io_dims = {
            "xpd": xpd_dims,"ypd": ypd_dims,
            "xdp": xdp_dims,"ydp": ydp_dims}

        return io_dims
    

    def get_io_sizes(self) -> dict[str,dict[str,int]]:
        """
        Get the output size of the network.

        Args:
            expanded:   If True, return the expanded size of the network.
                        If False, return the size of the network.

        Returns:
            io_sizes:    Dictionary of input/output sizes.
        """

        xpd_sizes = nh.get_io_sizes(self.io_idxs["xpd"])
        ypd_sizes = nh.get_io_sizes(self.io_idxs["ypd"])
        xdp_sizes = nh.get_io_sizes(self.io_idxs["xdp"])
        ydp_sizes = nh.get_io_sizes(self.io_idxs["ydp"])

        io_sizes = {
            "xpd": xpd_sizes,"ypd": ypd_sizes,
            "xdp": xdp_sizes,"ydp": ydp_sizes}

        return io_sizes
    
    @abstractmethod
    def forward(self, x) -> dict[str, torch.Tensor]:
        """
        Subclasses must implement a forward pass.
        """
        return None