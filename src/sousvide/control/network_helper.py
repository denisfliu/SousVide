import os
import json
import torch
import math

from typing import List,Union,Dict,Tuple,Any

def get_max_length(inputs: Dict[str,Dict[str,List[Union[str,int]]]]) -> int:
    """
    Get the maximum sequence length of the inputs.

    Args:
        inputs:         Dictionary of inputs.

    Returns:
        max_sequence:   Maximum sequence length.
    """

    max_frame = 0
    for input_key,input_value in inputs.items():
        if input_key == "history" or input_key == "feature":
            max_frame = max(max_frame,*input_value[0])

    max_length = max_frame + 1

    return max_length

def get_input_size(input_indices:List[List[torch.Tensor]]) -> List[int]:
    """
    Get the size of the input tensor.

    Args:
        input_indices:  Indices of the input.

    Returns:
        input_size:     Size of the input tensor.
    """

    input_size = math.prod(len(sublist) for sublist in input_indices)

    return input_size

def get_input_indices(inputs: Dict[str,Dict[str,List[Union[str,int]]]]) -> Dict[str,List[Union[slice,torch.Tensor]]] :
    """
    Extract the indices of the inputs. Inputs are defined to be one of the following:
        - rgb_image:    (Batch, Channels, Height, Width)
        - objective:    (Batch, Channels)
        - current:      (Batch, Channels)
        - history:      (Batch, Sequence, Channels)
        - feature:      (Batch, Sequence, Channels)
        - histNet:      (Batch, Channels)
        - featNet:      (Batch, Channels)
    The config jsons for input omit the batch dimension and use null to indicate elements
    that are numbered sequences (height, width, sequence). In the config jsons for pilots
    that call on these inputs, we use the string "all" to indicate all elements.

    Args:
        inputs:         Dictionary of inputs.

    Returns:
        indices:        Dictionary of input indices.
    """

    # Some useful paths
    workspace_path  = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    basic_config_path = os.path.join(
        workspace_path,"configs","inputs","basic.json")
    advd_config_path = os.path.join(
            workspace_path,"configs","inputs","advanced.json")

    # Load the basic and advanced configurations
    with open(basic_config_path) as f:
        basic_config = json.load(f)
    with open(advd_config_path) as f:
        advd_config = json.load(f)

    inputs_dict = {**basic_config,**advd_config}

    # Extract the indices
    indices:Dict[str,List[Union[slice,torch.Tensor]]] = {}
    for input,components in inputs.items():
        indices[input] = []
        input_list:list = inputs_dict[input][-1]
        sequences,channels = components[:-1],components[-1]

        for sequence in sequences:
            if sequence == ["all"]:
                indices[input].append(slice(None))
            else:
                indices[input].append(torch.tensor(sequence))

        if channels == ["all"]:
            indices[input].append(slice(None))
        else:
            channel_indices = [input_list.index(channel) for channel in channels]
            indices[input].append(torch.tensor(channel_indices))

        # Catch the case where the input is an rgb_image since convention
        # is actually (C,H,W) and not (H,W,C)
        if input == "rgb_image":
            indices[input] = indices[input][::-1]

    return indices

def extract_inputs(input:torch.Tensor, input_indices:List[Union[slice,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    """
    Extract the inputs from the input tensor. The first dimension of the input tensor is
    assumed to be the batch dimension and hence is left untouched.

    Args:
        input:          Input tensor.
        input_indices:  Indices of the inputs.

    Returns:
        xnn:    Extracted input tensor.

    """

    for dim, idx in enumerate(input_indices):
        if isinstance(idx, slice):
            indices = torch.arange(*idx.indices(input.shape[dim+1]))  # Convert slice to index list
        elif isinstance(idx,torch.Tensor):
            indices = idx
        else:
            raise ValueError(f"Invalid type in index_list[{dim}]: {type(idx)}. Must be slice or list.")
        # Move indices to the same device as the input tensor
        indices = indices.to(input.device)
 
        # Apply index selection along the current dimension
        input = torch.index_select(input, dim+1, indices)

    return input