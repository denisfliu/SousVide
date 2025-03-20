"""
Helper functions for trajectory data.
"""

import numpy as np
import numpy as np
import json
import os
import figs.utilities.trajectory_helper as th

from PIL import Image
from io import BytesIO
from typing import Dict,Union

def ts_to_obj(Tp:np.ndarray,CP:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory spline to an objective vector.

    Args:
        Tp:     Trajectory segment times.
        CP:     Trajectory control points.

    Returns:
        obj:    Objective vector.
    """
    tXU = th.TS_to_tXU(Tp,CP,None,1)

    return tXU_to_obj(tXU)

def tXU_to_obj(tXU:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory rollout to an objective vector.

    Args:
        tXU:    Trajectory rollout.

    Returns:
        obj:    Objective vector.
    """

    dt = tXU[0,-1]-tXU[0,0]
    dp = tXU[1:4,-1]-tXU[1:4,0]
    v0,v1 = tXU[4:7,0],tXU[4:7,-1]
    q0,q1 = tXU[7:11,0],tXU[7:11,-1]
    
    obj = np.hstack((dt,dp,v0,v1,q0,q1))

    return obj

def decompress_data(image_dict:Dict[str,Union[str,np.ndarray]]) -> Dict[str,Union[str,np.ndarray]]:
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    decompresses the images back to their original form.

    Args:
        image_dict:    Dictionary containing image data.

    Returns:
        image_dict:    Dictionary with decompressed images.
    """

    # Check if the image_dict has the key 'images' and if it is not empty
    assert 'images' in image_dict, "No images found in data dictionary"
    
    # Check if the image are processed or not. We do this by checking the array
    # order. If the order is (N, C, H, W) then the images are processed. If the
    # order is (N, H, W, C) then the images are unprocessed. We only compress if
    # the images are processed.
    if type(image_dict['images'][0]) == bytes:
        prc_imgs = image_dict['images']
        raw_imgs = []
        for frame_idx in range(len(prc_imgs)):
            # Process each image here
            imgpil = Image.open(BytesIO(prc_imgs[frame_idx]))
            raw_imgs.append(np.array(imgpil))

        image_dict['images'] = np.stack(raw_imgs,axis=0)
    else:
        pass

    return image_dict

def compress_data(Images):
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    compresses the images to a smaller size.

    Args:
        Images:    List of dictionaries containing image data.

    Returns:
        Images:    List of dictionaries with compressed images.
    """

    # Check if the Images list is not empty and contains dictionaries with 'images' key
    assert 'images' in Images[0], "No images found in data dictionary"

    for image_dict in Images:
        # Check if the image are processed or not. We do this by checking the array
        # order. If the order is (N, C, H, W) then the images are processed. If the
        # order is (N, H, W, C) then the images are unprocessed. We only compress if
        # the images are processed.

        if image_dict['images'].shape[-1] == 3:
            raw_imgs = image_dict['images']
            prc_imgs = []
            for frame_idx in range(raw_imgs.shape[0]):
                img_arr = raw_imgs[frame_idx]
                imgpil = Image.fromarray(img_arr)

                buffer = BytesIO()
                imgpil.save(buffer, format='PNG')   
                buffer.seek(0)
                prc_imgs.append(buffer.getvalue())

            image_dict['images'] = prc_imgs
        else:
            pass
    
    return Images

def load_config(config_name:str,config_type:str) -> Dict[str,Union[str,int,float]]:
    """
    Load the configuration for the specified config_name and config_type.

    Args:
        config_name:    Name of the configuration.
        config_type:    Type of the configuration (e.g., 'spline', 'rollout').

    Returns:
        config:         Dictionary containing the configuration.
    """
    
    # Get workspace path
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    config_path = os.path.join(
        workspace_path, 'configs', config_type, config_name + '.json')
    
    # Load configs
    with open(config_path) as json_file:
        config = json.load(json_file)

    return config

def compute_Tsp_batches(t0:float,tf:float,dt_ro:Union[None,float],
                        rate:Union[None,int],reps:Union[None,int],Nro_ds:int,
                        shuffle:bool=True) -> np.ndarray:
    """
    Compute the sample start times for a given rollout.

    Args:
        t0:     Start time of the trajectory.
        tf:     End time of the trajectory.
        dt_ro:  Sample time for each rollout.
        rate:   Number of time points per second.
        reps:   Number of rollouts per time point.
        Nro_ds: Number of rollouts per dataset.

    Returns:
        Tsp_bts:    Array containing batches of sampled start times.

    """
    
    # Compute the trajectory duration
    dt_tt = tf - t0                                            # Total trajectory duration
    
    # Catch None value cases
    dt_ro = dt_ro or dt_tt
    rate = rate or 1/dt_ro
    reps = reps or 1

    # Compute sample start times and batchify
    Ntp = int(rate*dt_tt)                                       # Number of time points per trajectory
    Nsp = int(reps*Ntp)                                         # Number of sample points (total)

    # Compute the sample points array
    if dt_ro == dt_tt:
        Tsp = np.array([t0]*Nsp)                                # Entire sample points array
    else:
        Tsp = np.tile(np.linspace(t0,tf,Ntp+1)[:-1],reps)     # Entire sample points array
        Tsp += np.random.uniform(-1/rate,1/rate,Nsp)        # Add some noise to the sample points array
        Tsp = np.clip(Tsp,t0,tf)                                # Clip the sample points array

    # Shuffle if required
    if shuffle:
        np.random.shuffle(Tsp)

    # Split into batches of datasets
    Tsp_bts = np.split(Tsp,np.arange(Nro_ds,Nsp,Nro_ds))        # Split the sample points array into their batches

    return Tsp_bts