import os
import json
import torch
import math

from typing import List,Union,Dict,Literal

def get_max_length(io_idxs: Dict[str,List[torch.Tensor]]) -> int:
    """
    Get the maximum sequence length of the inputs/outputs.

    Args:
        io_idxs:    Dictionary of input/output indicies

    Returns:
        max_seq:    Myaximum sequence length.
    """
    io_seq_refr = get_io_refr("sequence")

    max_frame = 0
    for name,idxs in io_idxs.items():
        if name in io_seq_refr:
            frames = idxs[-2]
            max_frame = max(max_frame,*frames)

    max_seq = max_frame + 1

    return max_seq

def get_io_refr(io_type:Literal["basic","sequence","image"]) -> Dict[str, List[List[Union[str,int]]]]:
    """
    Get the input/output reference dictionary.

    Returns:
        io_refr:    Input/output dictionary.
    """

    # Some useful paths
    workspace_path  = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    config_path = os.path.join(
                workspace_path,"configs","nnio",f"{io_type}.json")
    
    # Get the input/output reference dictionary
    with open(config_path) as f:
        io_refr = json.load(f)

    return io_refr

def get_io_size(io_idxs:Dict[str,List[Union[slice,torch.Tensor]]]) -> int:
    """
    Get the size of the input/output.

    Args:
        idxs_dict:  Dictionary of indices of the input/output.

    Returns:
        io_size:    Size of the input/output tensor.
    """

    io_size = 0
    for idxs_list in io_idxs.values():
        io_size += math.prod(len(sublist) for sublist in idxs_list)

    return io_size

def get_io_idxs(io_cfgs: Dict[str, List[List[Union[int, str]]]]) -> Dict[str,List[Union[slice,torch.Tensor]]] :
    """
    Extract the indices of the inputs/outputs. Inputs/outputs are defined to be one of the following:
        
    Basic
        - objective:    (Batch, Channels)
        - current:      (Batch, Channels)
        - command:      (Batch, Channels)
        - parameters:   (Batch, Channels)
        - histLat:      (Batch, Channels)
        - featLat:      (Batch, Channels)
    Sequence
        - history:      (Batch, Window, Channels)
        - feature:      (Batch, Window, Channels)
        - forces:       (Batch, Window, Channels)
    Image
        - rgb_image:    (Batch, Height, Width, Channels)

    The config jsons for input/output omit the batch dimension and use null to indicate elements that
    are numbered sequences (height, width, window). In the config jsons for pilots that call on these
    inputs/outputs, we use <int>/["all"] to refer to the total input/output size and List[int] to refer
    to specific indices of the input/output.

    Args:
        io_cfgs: Config dictionary of inputs/outputs.

    Returns:
        io_idxs:   Dictionary of input/output indices.
    """

    # Get the input/output dictionary
    io_bsc_refr = get_io_refr("basic")
    io_seq_refr = get_io_refr("sequence")
    io_img_refr = get_io_refr("image")
    io_refr = {**io_bsc_refr, **io_seq_refr, **io_img_refr}
    
    # Extract the indices
    io_idxs = {}
    for io,config in io_cfgs.items():
        # Extract the reference (accounting for numbered io)
        if io[-1].isdigit():
            refr = io_refr[io[:-1]][-1]
        else:
            refr = io_refr[io][-1]

        sequences,channels = config[:-1],config[-1]

        idxs = []
        for sequence in sequences:
            if sequence == ["all"]:
                idxs.append(slice(None))
            else:
                idxs.append(torch.tensor(sequence))

        if channels == ["all"]:
            idxs.append(slice(None))
        else:
            channel_indices = [refr.index(channel) for channel in channels]
            idxs.append(torch.tensor(channel_indices))

        # Catch the case where the io is an rgb_image since convention
        # is actually (C,H,W) and not (H,W,C)
        if io == "rgb_image":
            idxs = idxs[::-1]

        # Store the indices
        io_idxs[io] = idxs

    return io_idxs

def extract_io(io_srcs:Dict[str,torch.Tensor],
               io_idxs:Union[None,Dict[str,List[Union[slice,torch.Tensor]]]],
               use_tensor:bool=False,flatten:bool=False) -> Dict[str,List[torch.Tensor]]:
    """
    Extract the inputs/outputs from the input/output tensor. The first dimension of the tensor is
    assumed to be the batch dimension and hence is left untouched.

    Args:
        io_dict:    Input/Output dictionary tensor.
        io_idxs:    Dictionary of indices of the inputs/outputs.
        use_tensor: If True, return a tensor of the inputs/outputs. If False, return a list.
        flatten:    If True, flatten the non-batch dimensions of the inputs/outputs.

    Returns:
        xnn:        List/tensor of extracted inputs.
    """

    # Get the input/output tensors from the input/out dictionary
    if io_idxs is None:
        # Return all if no indices are provided
        xnn = list(io_srcs.values())
    else:
        xnn = []
        for name,idxs in io_idxs.items():
            # Remove last letter of the name if it is a number
            if name[-1].isdigit():
                name = name[:-1]

            # Extract the input/output tensor
            data = io_srcs[name]
            for dim, idx in enumerate(idxs):
                if isinstance(idx, slice):
                    idxs = torch.arange(*idx.indices(data.shape[dim+1]))
                elif isinstance(idx,torch.Tensor):
                    idxs = idx
                else:
                    raise ValueError(f"Invalid type in index_list[{dim}]: {type(idx)}. Must be slice or list.")
                
                # Move indices to the same device as the input/output tensor
                idxs = idxs.to(data.device)
        
                # Apply index selection along the current dimension
                data = torch.index_select(data, dim+1, idxs)

            xnn.append(data)

    # Convert to tensor if requested
    if use_tensor:
        xnn = torch.cat(xnn)

    # Flatten the non-batch dimensions if requested
    if flatten:
        xnn = torch.flatten(xnn, start_dim=1)

    return xnn

def generate_positional_encoding( d_model, max_seq_len):
    """
    Generate positional encoding.

    Args:
        d_model:        Dimension of the model.
        max_seq_len:    Maximum sequence length.

    Returns:
        pe:             Positional encoding.

    """

    # Generate the positional encoding
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)