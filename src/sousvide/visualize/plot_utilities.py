import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import figs.utilities.transform_helper as th
import figs.utilities.orientation_helper as oh

from typing import Tuple,List

def get_plot_limits(XX:List[np.ndarray],
                    lim_min:bool=True,lim_max:bool=True,abs_max:float=10.0) -> Tuple[np.ndarray,np.ndarray]:
    """
    Get the plot limits for the trajectory.
    """

    x_dim = XX[0].shape[-1]
    if lim_min == True:
        x_max,x_min = np.ones(x_dim),-np.ones(x_dim)
    else:
        x_max,x_min = np.zeros(x_dim),np.zeros(x_dim)

    for X in XX:
        xi_min = np.min(X,axis=0)
        xi_max = np.max(X,axis=0)

        x_min = np.minimum(x_min,xi_min)
        x_max = np.maximum(x_max,xi_max)

    if lim_max == True:
        x_max = np.minimum(x_max,abs_max)
        x_min = np.maximum(x_min,-abs_max)

    # Pad the limits for better visualization
    delta = 0.1*(x_max-x_min)
    x_min -= delta
    x_max += delta
    
    return x_min,x_max

def quad_frame(x:np.ndarray,ax:plt.Axes,scale:float=1.0):
    """
    Plot a quadcopter frame in 3D.
    """
    frame_body = scale*np.diag([0.6,0.6,-0.2])
    frame_labels = ["red","green","blue"]
    pos  = x[0:3]
    quat = x[6:10]
    
    for j in range(0,3):
        Rj = R.from_quat(quat).as_matrix()
        arm = Rj@frame_body[j,:]

        frame = np.zeros((3,2))
        if (j == 2):
            frame[:,0] = pos
        else:
            frame[:,0] = pos - arm

        frame[:,1] = pos + arm

        ax.plot(frame[0,:],frame[1,:],frame[2,:], frame_labels[j],label='_nolegend_')
