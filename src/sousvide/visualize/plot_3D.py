import numpy as np
import matplotlib.pyplot as plt
import sousvide.visualize.plot_utilities as pu

from typing import Dict,Union,Tuple,List

def tXU_to_3D(tXU_list:List[np.ndarray],
              WPs:np.ndarray=None,tXUd:np.ndarray=None,
              scale:float=1.0,n:int=500,plot_last:bool=False):
    
    # Compute some useful variables
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]
    tX_min,tX_max = pu.get_plot_limits(tXU_list,use_deadzone=True)
    
    plim = np.array([
        [ tX_min[1], tX_max[1]],
        [ tX_min[2], tX_max[2]],
        [  0.0, -3.0]])    

    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    # Initialize World Frame Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()
    ax.invert_yaxis()

    # Rollout the world frame trajectory
    for idx,tXU in enumerate(tXU_list):
        # Plot the world frame trajectory
        ax.plot(tXU[1,:], tXU[2,:], tXU[3,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        for i in range(0,tXU.shape[1],n):
            pu.quad_frame(tXU[1:11,i],ax,scale=scale)

        if plot_last == True or idx == 0:
            pu.quad_frame(tXU[1:11,-1],ax,scale=scale)

    # Plot the control points if provided
    if WPs is not None:
        for i in range(WPs.shape[1]):
            ax.scatter(WPs[0,i],WPs[1,i],WPs[2,i],color=traj_colors[idx%len(traj_colors)],marker='x')

    # Plot the desired trajectory if provided
    if tXUd is not None:
        ax.plot(tXUd[1,:], tXUd[2,:], tXUd[3,:],color='k', linestyle='--',linewidth=0.8)

    plt.show(block=False)

def CP_to_3D(Tp:List[np.ndarray],CP:List[np.ndarray],
                hz:int=20,scale:float=1.0,n:int=500,plot_last:bool=False):

    """"
    Plot the trajectory in 3D space from control point rollout."
    """

    # Unpack the trajectory
    tXU_list:List[np.ndarray] = []
    WPs = np.zeros((3,len(Tp)))
    for i in range(len(Tp)):
        T,X = pu.unpack_trajectory(Tp[i],CP[i],hz)
        U = np.zeros((3,X.shape[1]))

        tXU = np.vstack((T,X,U))
        tXU_list.append(tXU)
        WPs[0,i] = CP[i][0,0]
        WPs[1,i] = CP[i][1,0]
        WPs[2,i] = CP[i][2,0]

    # Plot the trajectory
    tXU_to_3D(tXU_list,WPs=WPs,scale=scale,n=n,plot_last=plot_last)

def RO_to_3D(RO:List[Dict[str,Union[np.ndarray,int]]],
              n:int=500,scale=1.0,plot_last:bool=False):

    # Unpack the trajectory
    tXU_list:List[np.ndarray] = []
    for idx,ro in enumerate(RO):
        Tro = ro["Tro"]
        Xro = ro["Xro"]
        Uro = ro["Uro"]

        # Ensure the trajectory is of the same length
        upad = np.zeros((Uro.shape[0],1))
        Uro = np.hstack((Uro,upad))

        tXU = np.vstack((Tro,Xro,Uro))
        tXU_list.append(tXU)

    # Get the desired trajectory if available
    if "tXd" in RO[0]:
        tXd = RO[0]["tXd"]
        tXUd = np.vstack((tXd,np.zeros((4,tXd.shape[1]))))
    else:
        tXUd = None
    
    # Plot the trajectory
    tXU_to_3D(tXU_list,tXUd=tXUd,scale=scale,n=n,plot_last=plot_last)
