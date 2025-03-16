"""
Helper functions for trajectory data.
"""

import numpy as np
import numpy as np
import torch
import figs.utilities.trajectory_helper as th

from PIL import Image
from io import BytesIO
from typing import Dict,Union,List

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
    """
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
    """
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