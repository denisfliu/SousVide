import torch
import sousvide.control.network_factory as nf
import sousvide.control.network_helper as nh

from torch import nn
from sousvide.control.networks.base_net import BaseNet

class Policy(nn.Module):
    def __init__(self,
                 policy_config:dict[str,dict],
                 policy_name:str,
                 policy_path:str):
        """
        Initialize a Learned Control Policy.

        Args:
            policy_config:  Policy configuration dictionary.
            policy_path:    Policy path.
            
        Variables:
            network_type:   Type of network.
            fpass_indices:  Indices of the forward-pass output.
            label_indices:  Indices of the label output.
            networks:       Network layers.
            use_fpass:      Flag to use forward-pass.
            nhy:            Maximum sequence length.
        """
        
        # Initial Parent Call
        super().__init__()

        # Populate the network
        networks:dict[str,BaseNet] = nn.ModuleDict()
        nhy = 1
        for name,config in policy_config["networks"].items():
            networks[name],nhy_net = nf.generate_network(config,name,policy_path)   

            # Update the max sequence length variable
            if nhy_net is not None:
                nhy = max(nhy,nhy_net)

            # Ensure all fpass flags are true
            for network in networks.values():
                network.use_fpass = True

        # Class Variables (last network outputs command)
        self.fpass_indices = network.fpass_indices
        self.label_indices = network.label_indices
        self.network_type = policy_name
        self.use_fpass = True
        self.nhy = int(nhy)

        self.networks = networks

    def forward(self,
                xnn_im:torch.Tensor,xnn_ob:torch.Tensor,
                xnn_cr:torch.Tensor,xnn_hy:torch.Tensor,xnn_ft:torch.Tensor) -> tuple[
                    torch.Tensor,torch.Tensor,dict[str,list[torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            xnn_im: RGB Image input.
            xnn_ob: Objective input.
            xnn_cr: Current input.
            xnn_hy: History input.
            xnn_ft: Feature input.

        Returns:
            ynn:        Policy output.
            znn:        Feature output.
            xnn_dict:   Dictionary of Network inputs.
        """

        # Initialize the input source dictionary
        xnn_srcs = {
            "rgb_image": xnn_im, "objective": xnn_ob,
            "current": xnn_cr, "history": xnn_hy, "features": xnn_ft
        }

        # Initialize the function outputs
        ynn,znn = torch.zeros(4),{}
        xnn_dict = {}

        # Forward Pass through the networks
        for net_name,network in self.networks.items():
            # Extract the Network Inputs and Input Key
            xnn_net = nh.extract_io(xnn_srcs,network.input_indices)
            xnn_key = next(iter(network.fpass_indices))

            # Forward Pass through the Network
            ynn_net = network(*xnn_net)
            
            # Update xnn_srcs
            xnn_srcs[xnn_key] = ynn_net

            # Update the znn dictionary
            znn[net_name] = ynn_net

            # Store input data for training
            xnn_dict[net_name] = xnn_net

        # Last network output is the policy output
        ynn = ynn_net

        if self.use_fpass:
            return ynn,znn,xnn_dict
        else:
            return ynn