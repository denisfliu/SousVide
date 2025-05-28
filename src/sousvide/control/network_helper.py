import os
import json
import torch
import math

import sousvide.visualize.rich_utilities as ru

from typing import Literal

def get_max_length(io_idxs: dict[str,list[torch.Tensor]]) -> int:
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

def get_io_refr(io_type:Literal["basic","sequence","image"]) -> dict[str, list[list[str|int]]]:
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

def get_io_dims(io_idxs:dict[str,list[slice|torch.Tensor]]) -> dict[str,list[int]]:
    """
    Get the dimensions of the input/output.

    Args:
        io_idxs:  Dictionary of indices of the input/output.

    Returns:
        io_dims:   Dimensions of the input/output tensors.
    """

    io_dims = {}
    for io_name,io_idxs in io_idxs.items():
        io_dim = [len(sublist) for sublist in io_idxs]
        io_dims[io_name] = io_dim

    return io_dims

def get_io_sizes(io_idxs:dict[str,list[slice|torch.Tensor]]) -> dict[str,list[int]]:
    """
    Get the total size of the input/output.

    Args:
        io_idxs:  Dictionary of indices of the input/output.

    Returns:
        io_sizes:   Sizes of the input/output tensors.
    """
    # Get the input/output dimensions
    io_dims = get_io_dims(io_idxs)

    io_sizes = {}
    for io_name,io_dim in io_dims.items():
        io_sizes[io_name] = math.prod(io_dim)

    return io_sizes

def get_io_idxs(io_cfgs: dict[str, list[list[int|str]]]) -> dict[str,list[slice|torch.Tensor]] :
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

        idxs = []
        for dim in config:
            if isinstance(dim,int):
                idxs.append(torch.arange(dim))
            elif isinstance(dim,list) and all(isinstance(i, int) for i in dim):
                idxs.append(torch.tensor(dim))
            elif isinstance(dim,list) and all(isinstance(i, str) for i in dim):
                channel_indices = [refr.index(channel) for channel in dim]
                idxs.append(torch.tensor(channel_indices))

        # Catch the case where the io is an rgb_image since convention
        # is actually (C,H,W) and not (H,W,C)
        if io == "rgb_image":
            idxs = idxs[::-1]

        # Store the indices
        io_idxs[io] = idxs

    return io_idxs

def extract_io(io_srcs:dict[str,torch.Tensor],
               io_idxs:None|dict[str,list[slice|torch.Tensor]],
               use_tensor:bool=False,flatten:bool=False) -> dict[str,list[torch.Tensor]]:
    """
    Extract the inputs/outputs from the input/output tensor. The first dimension of the tensor is
    assumed to be the batch dimension and hence is left untouched.

    Args:
        io_dict:    Input/Output dictionary tensor.
        io_idxs:    Dictionary of indices of the inputs/outputs.

    Returns:
        xnn:        Dictionary of extracted inputs.
    """

    xnn = {}
    for name,idxs in io_idxs.items():
        # Remove last letter of the name if it is a number
        if name[-1].isdigit():
            name_dt = name[:-1]
        else:
            name_dt = name

        # Extract the input/output tensor
        data = io_srcs[name_dt]
        for dim, idx in enumerate(idxs):
            if isinstance(idx,torch.Tensor):
                idxs = idx
            else:
                raise ValueError(f"Invalid type in index_list[{dim}]: {type(idx)}. Must be slice or list.")

            # Move indices to the same device as the input/output tensor
            idxs = idxs.to(data.device)

            # Apply index selection along the current dimension
            data = torch.index_select(data, dim+1, idxs)

        xnn[name] = data

    return xnn

def generate_positional_encoding(d_model, max_seq_len=100):
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