"""
Helper functions for sous vide.
"""

import numpy as np
import numpy as np
import json
import os
import figs.utilities.trajectory_helper as th

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
