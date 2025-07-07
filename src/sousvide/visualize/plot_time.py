import numpy as np
import matplotlib.pyplot as plt
import sousvide.visualize.plot_utilities as pu

from typing import Dict,Union,List

def tXU_to_time(tXU_list:List[np.ndarray],tXd:np.ndarray=None):
    
    # Compute some useful variables
    pv_labels = [["$p_x$","$p_y$","$p_z$"],["$v_x$","$v_y$","$v_z$"]]
    qu_labels = [[r"$q_x$", r"$q_y$", r"$q_z$", "q_w"], [r"$f_{th}$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]]

    tXU_min,tXU_max = pu.get_plot_limits(tXU_list,lim_min=True)
   
    # Plot Positions and Velocities
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            idd = j+(3*i)+1

            for idx,tXU in enumerate(tXU_list):
                axs[j,i].plot(tXU[:,0],tXU[:,idd],alpha=0.5)

            axs[j,i].set_ylim((tXU_min[idd],tXU_max[idd]))
            axs[j,i].set_ylabel(pv_labels[i][j])

            if tXd is not None:
                axs[j,i].plot(tXd[:,0],tXd[:,idd],color='k', linestyle='--',linewidth=0.8)

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    ref, = axs[0, 0].plot([], [], 'k--', label='reference')
    fig.legend(handles=[ref],loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the subplot layout to make room for the legend
    plt.show()

    # Plot Orientation and Body Rates
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(4):
            if not ((i==1) and (j==3)):
                idd = 7+j+(4*i)

            for idx,tXU in enumerate(tXU_list):
                axs[j,i].plot(tXU[:,0],tXU[:,idd],alpha=0.5)
                
            axs[j,i].set_ylim((tXU_min[idd],tXU_max[idd]))
            axs[j,i].set_ylabel(qu_labels[i][j])

            if tXd is not None:
                if i == 0:
                    axs[j,i].plot(tXd[:,0],tXd[:,idd],color='k', linestyle='--',linewidth=0.8)

    axs[0, 1].invert_yaxis()
    axs[0, 0].set_title('Orientation')
    axs[0, 1].set_title('Control Inputs')
    ref, = axs[0, 0].plot([], [], 'k--', label='reference')
    fig.legend(handles=[ref],loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the subplot layout to make room for the legend
    plt.show(block=False)

def RO_to_time(RO:List[Dict[str,Union[np.ndarray,int]]]):

    # Unpack the trajectory
    tXU_list:List[np.ndarray] = []
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
    else:
        tXd = None
    
    # Plot the trajectory
    tXU_to_time(tXU_list,tXd=tXd)
