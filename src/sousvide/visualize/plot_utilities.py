import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import figs.utilities.trajectory_helper as th

from typing import Tuple,List

def get_plot_limits(XX:List[np.ndarray],use_deadzone:bool=True) -> Tuple[np.ndarray,np.ndarray]:
    """
    Get the plot limits for the trajectory.
    """
    x_dim = XX[0].shape[0]
    if use_deadzone == True:
        x_max,x_min = np.ones(x_dim),-np.ones(x_dim)
    else:
        x_max,x_min = np.zeros(x_dim),np.zeros(x_dim)

    for X in XX:
        xi_min = np.min(X,axis=1)
        xi_max = np.max(X,axis=1)

        x_min = np.minimum(x_min,xi_min)
        x_max = np.maximum(x_max,xi_max)

    # Pad the limits for better visualization
    delta = 0.1*(x_max-x_min)
    x_min -= delta
    x_max += delta
    
    return x_min,x_max

def unpack_trajectory(Tp:np.ndarray,CP:np.ndarray,hz:int,
                      mode:str='time',trim:bool=True) -> Tuple[np.ndarray,np.ndarray]:
    """
    Unpack the trajectory from the control points.
    """

    # Unpack the trajectory
    Nt = int(Tp[-1]*hz+1)
    T = np.zeros(Nt)

    if mode == 'time':
        X = np.zeros((13,Nt))
    else:
        X = np.zeros((Nt,4,12))
        
    idx = 0
    for k in range(Nt):
        tk = Tp[0]+k/hz

        if trim == True:
            tk = np.minimum(Tp[-1],tk)
        if tk > Tp[idx+1] and idx < len(Tp)-2:
            idx += 1

        t0,tf = Tp[idx],Tp[idx+1]
        fo = th.ts_to_fo(tk-t0,tf-t0,CP[idx,:,:])

        T[k] = tk
        if mode == 'time':
            X[:,k] = th.fo_to_xu(fo)[0:13]
        else:
            X[k,:,:] = fo

    # Ensure continuity of the quaternion
    if mode == 'time':
        qr = np.array([0.0,0.0,0.0,1.0])
        for k in range(Nt):
            q = X[6:10,k]
            qc = th.obedient_quaternion(q,qr)

            X[6:10,k] = qc
            qr = qc
        
    return T,X

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
