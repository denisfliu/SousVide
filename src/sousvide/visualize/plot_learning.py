from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal
import os
import torch
import sousvide.visualize.plot_synthesize as ps
from tabulate import tabulate

def plot_losses(cohort_name:str,roster:List[str],topic:str):
    """
    Plot the losses for each student in the roster.
    """

    # # Clear all plots
    # plt.close('all')

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)

    # Print the header
    print("=====================================================")
    print("Cohort: ",cohort_name)
    print("Topic : ",topic)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    # Plot the losses for each student
    for student_name in roster:
        try:
            student_path = os.path.join(cohort_path,"roster",student_name)
            losses_path = os.path.join(student_path,"losses_"+topic+".pt")

            losses:dict = torch.load(losses_path)
        except:
            print("-----------------------------------------------------")
            print(f"Student {student_name} did not take {topic}.")
            print("-----------------------------------------------------")
            continue
        
        # Gather plot data
        Loss_tn,Loss_tt,Neps = [],[],0
        active_net,update_nets = [],[]
        Nd_tn,Nd_tt = [],[]
        T_tn = 0
        for loss_data in losses.values():
            Loss_tn.append(loss_data["Loss_tn"])
            Loss_tt.append(loss_data["Loss_tt"])

            Neps += loss_data["N_eps"]

            Nd_tn.append(loss_data["Nd_tn"])
            Nd_tt.append(loss_data["Nd_tt"])

            T_tn += loss_data["t_tn"]

        Loss_tn = np.hstack(Loss_tn)
        Loss_tt = np.hstack(Loss_tt)

        # Compute the training time
        hours = T_tn // 3600
        minutes = (T_tn % 3600) // 60
        seconds = np.around(T_tn % 60,1)

        # Print some overall stuff
        print("-----------------------------------------------------")
        print(f"Student:{student_name}")
        print(f"Train/Test Loss: {np.around(Loss_tn[-1],3)}/{np.around(Loss_tt[-1],3)}")
        print(f"Train/Test Size: {Nd_tn[-1]}/{Nd_tt[-1]}")
        print(f"Total Epochs: {Neps}")
        print(f"Training Time: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")

        axs[0].plot(Loss_tn,label=student_name)
        axs[1].plot(Loss_tt,label=student_name)

    axs[0].set_title('Training Loss')
    axs[0].legend(loc='upper right')

    axs[1].set_title('Testing Loss')
    axs[1].legend(loc='upper right')

    # Set common labels
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Loss')

    # # Set the y-axis limits
    # axs[0].set_ylim([0, 0.06])
    # axs[1].set_ylim([0, 0.06])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show(block=False)

def review_simulations(cohort_name:str,course_name:str,roster:List[str],plot_show:bool=False):
    """
    Plot the simulations for each student in the roster.

    """
    
    # Add Expert to Roster
    roster = ["expert"]+roster

    # Initialize Table for plotting and visualization
    headers=["Mean Solve (Hz)","Worst Solve (Hz)",
             "Pos TTE (m)","Best Pos TTE (m)"]
    table = []

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_path = os.path.join(workspace_path,"cohorts",cohort_name,"output")

    # Print some overall stuff
    print("========================================================================================================")
    print("Cohort Name:",cohort_name)
    print("Course Name:",course_name)

    print("Roster:",roster)
    for pilot_name in roster:
        # Load Simulation Data
        trajectories = torch.load(os.path.join(output_path,"sim_"+course_name+"_"+pilot_name+".pt"))
        
        Ebnd = np.zeros((len(trajectories),trajectories[0]["Ndata"]))
        Tsol = np.zeros((len(trajectories),trajectories[0]["Ndata"]))
        methods = []
        for idx,trajectory in enumerate(trajectories):
            # Extract Method Name
            method_name = trajectory["rollout_id"].split("_")[0]
            if method_name not in methods:
                methods.append(method_name)

            # Error Bounds
            for i in range(trajectory["Ndata"]):
                Ebnd[idx,i] = np.min(np.linalg.norm(trajectory["Xro"][:,i].reshape(-1,1)-trajectory["tXUd"][1:11,:],axis=0))

            # Total Solve Time
            Tsol[idx,:] = np.sum(trajectory["Tsol"],axis=0)

        # Trajectory Data
        pilot_name = pilot_name
        mean_solve = 1/np.mean(Tsol)
        worst_solve = 1/np.max(Tsol)
        mean_error = np.mean(Ebnd)
        mean_error_traj = np.mean(Ebnd,axis=1)
        best_error_idx = np.argmin(mean_error_traj)
        best_error = mean_error_traj[best_error_idx]

        # Append Data
        table.append([pilot_name,mean_solve,worst_solve,mean_error,best_error])

        if plot_show:
            print("========================================================================================================")
            print("Visualization ------------------------------------------------------------------------------------------")
            print("Pilot Name    :",pilot_name)
            print("Test Method(s):",methods)
            ps.RO_to_spatial(trajectories,plot_last=True,tXUd=trajectories[0]["tXUd"])

    print("========================================================================================================")
    print("Performance --------------------------------------------------------------------------------------------")
    print(tabulate(table,headers=headers,tablefmt="grid"))
