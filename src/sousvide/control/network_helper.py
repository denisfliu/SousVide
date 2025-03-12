import os
import json
import torch
import math

from typing import List,Union,Dict

def get_max_length(inputs: Dict[str,Dict[str,List[Union[str,int]]]]) -> int:
    """
    Get the maximum sequence length of the inputs/outputs.

    Args:
        inputs:         Dictionary of inputs/outputs.

    Returns:
        max_sequence:   Maximum sequence length.
    """

    max_frame = 0
    for input_key,input_value in inputs.items():
        if input_key == "history" or input_key == "feature":
            max_frame = max(max_frame,*input_value[0])

    max_length = max_frame + 1

    return max_length

def get_io_dict() -> Dict[str, List[List[str]]]:
    """
    Get the input/output dictionary.

    Returns:
        io_dict:    Input/output dictionary.
    """

    # Some useful paths
    workspace_path  = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    basic_config_path = os.path.join(
        workspace_path,"configs","nnio","basic.json")
    seq_config_path = os.path.join(
        workspace_path,"configs","nnio","sequence.json")

    # Load the basic and advanced configurations
    with open(basic_config_path) as f:
        basic_config = json.load(f)
    with open(seq_config_path) as f:
        seq_config = json.load(f)

    io_dict = {**basic_config,**seq_config}

    return io_dict

def get_io_size(idxs_dict:Dict[str,List[Union[slice,torch.Tensor]]]) -> int:
    """
    Get the size of the input/output.

    Args:
        idxs_dict:  Dictionary of indices of the input/output.

    Returns:
        io_size:    Size of the input/output tensor.
    """

    io_size = 0
    for idxs_list in idxs_dict.values():
        io_size += math.prod(len(sublist) for sublist in idxs_list)

    return io_size

def get_io_dims(idx_list:List[Union[slice,torch.Tensor]]) -> int:
    """
    Get the dimensions of the input/output.

    Args:
        idxs:       Indices of the input/output.

    Returns:
        io_dims:    Dimensions of the input/output tensor.
    """

    io_dims = [len(sublist) for sublist in idx_list]

    return io_dims

def get_io_indices(ios: Dict[str, List[List[Union[int, str]]]]) -> Dict[str,List[Union[slice,torch.Tensor]]] :
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
        - rgb_image:    (Batch, Height, Width, Channels)
        - history:      (Batch, Window, Channels)
        - feature:      (Batch, Window, Channels)
        - forces:       (Batch, Window, Channels)

    The config jsons for input/output omit the batch dimension and use null to indicate elements that
    are numbered sequences (height, width, window). In the config jsons for pilots that call on these
    inputs/outputs, we use <int>/["all"] to refer to the total input/output size and List[int] to refer
    to specific indices of the input/output.

    Args:
        ios:        Dictionary of inputs/outputs.

    Returns:
        idxs:       Dictionary of input/output indices.
    """

    # Get the input/output dictionary
    io_dict = get_io_dict()
    
    # Extract the indices
    idxs:Dict[str,List[Union[slice,torch.Tensor]]] = {}
    for io,components in ios.items():
        idxs[io] = []
        io_key = io_dict[io][-1]
        sequences,channels = components[:-1],components[-1]

        for sequence in sequences:
            if sequence == ["all"]:
                idxs[io].append(slice(None))
            else:
                idxs[io].append(torch.tensor(sequence))

        if channels == ["all"]:
            idxs[io].append(slice(None))
        else:
            channel_indices = [io_key.index(channel) for channel in channels]
            idxs[io].append(torch.tensor(channel_indices))

        # Catch the case where the io is an rgb_image since convention
        # is actually (C,H,W) and not (H,W,C)
        if io == "rgb_image":
            idxs[io] = idxs[io][::-1]

    return idxs

def extract_io(io:torch.Tensor, io_idxs:List[Union[slice,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    """
    Extract the inputs/outputs from the input/output tensor. The first dimension of the tensor is
    assumed to be the batch dimension and hence is left untouched.

    Args:
        input:          Input tensor.
        input_indices:  Indices of the inputs.

    Returns:
        xnn:    Extracted input tensor.

    """

    for dim, idx in enumerate(io_idxs):
        if isinstance(idx, slice):
            idxs = torch.arange(*idx.indices(io.shape[dim+1]))  # Convert slice to index list
        elif isinstance(idx,torch.Tensor):
            idxs = idx
        else:
            raise ValueError(f"Invalid type in index_list[{dim}]: {type(idx)}. Must be slice or list.")
        
        # Move indices to the same device as the input/output tensor
        idxs = idxs.to(io.device)
 
        # Apply index selection along the current dimension
        io = torch.index_select(io, dim+1, idxs)

    return io