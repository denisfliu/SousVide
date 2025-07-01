import numpy as np
import matplotlib.pyplot as plt
import sousvide.visualize.plot_utilities as pu

def tXU_to_3D(tXU_list:list[np.ndarray],
              WPs:np.ndarray=None,tXUd:np.ndarray=None,
              scale:float=0.5,n:int=500,plot_last:bool=False):
    
    # Compute some useful variables
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]
    tX_min,tX_max = pu.get_plot_limits(tXU_list,lim_min=True,lim_max=True)
    
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
        ax.plot(tXU[:,1], tXU[:,2], tXU[:,3],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        for i in range(0,tXU.shape[1],n):
            pu.quad_frame(tXU[i,1:11],ax,scale=scale)

        if plot_last == True:
            pu.quad_frame(tXU[-1,1:11],ax,scale=scale)

    # Plot the control points if provided
    if WPs is not None:
        for i in range(WPs.shape[0]):
            ax.scatter(WPs[i,0],WPs[i,1],WPs[i,2],color=traj_colors[idx%len(traj_colors)],marker='x')

    # Plot the desired trajectory if provided
    if tXUd is not None:
        ax.plot(tXUd[:,1], tXUd[:,2], tXUd[:,3],color='k', linestyle='--',linewidth=0.8)

    plt.show(block=False)

def RO_to_3D(RO:list[dict[str,np.ndarray|int]],
              n:int=500,scale=1.0,plot_last:bool=False):

    # Unpack the trajectory
    tXU_list:list[np.ndarray] = []
    for ro in RO:
        Tro = ro["Tro"]
        Xro = ro["Xro"]
        Uro = ro["Uro"]

        # Ensure the trajectory is of the same length
        upad = np.zeros(Uro.shape[-1])
        Uro = np.vstack((Uro,upad))

        tXU = np.hstack((Tro.reshape((-1,1)),Xro,Uro))
        tXU_list.append(tXU)

    # Get the desired trajectory if available
    if "tXd" in RO[0]:
        tXd = RO[0]["tXd"]
        tXUd = np.hstack((tXd,np.zeros((tXd.shape[0],4))))
    else:
        tXUd = None
    
    # Plot the trajectory
    tXU_to_3D(tXU_list,tXUd=tXUd,scale=scale,n=n,plot_last=plot_last)
