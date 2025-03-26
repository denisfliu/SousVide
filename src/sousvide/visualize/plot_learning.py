import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import sousvide.visualize.plot_3D as p3
import sousvide.visualize.rich_utilities as ru
import sousvide.flight.flight_helper as fh

from typing import List

def plot_losses(cohort_name:str, roster:List[str], network_name:str, Nln:int=70):
    """
    Plot the losses for each student in the roster.
    """

    # Initialize the rich variables
    console = ru.get_console()

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)

    # Compile the learning summary header
    learning_summary = [
        f"{'=' * Nln}\n"
        f"Cohort : [bold cyan]{cohort_name}[/]\n"
        f"Network: [bold cyan]{network_name}[/]\n"
        f"{'=' * Nln}"]

    # Plot the losses for each student
    roster_data,Nplot,xplim = {},0,0
    for student_name in roster:
        try:
            student_path = os.path.join(cohort_path, "roster", student_name)
            losses_path = os.path.join(student_path, f"losses_{network_name}.pt")

            losses: dict = torch.load(losses_path)
        except:
            console.print(f"{'-' * Nln}\n"
                          f"Student [bold cyan]{student_name}[/bold cyan] does not have a [bold cyan]{network_name}[/bold cyan].")
            continue

        # Gather plot data
        Loss_tn, Loss_tt, Eval_tte, Neps = [], [], [], []
        Nd_tn, Nd_tt, T_tn = [], [], []
        for loss_data in losses.values():
            # Add the loss data to the lists
            Loss_tn.append(loss_data["Loss_tn"])
            Loss_tt.append(loss_data["Loss_tt"])
            Eval_tte.append(loss_data["Eval_tte"])

            # Update the total number of episodes and other metrics
            Neps.append(loss_data["N_eps"])

            # Append the number of data points for training and testing
            Nd_tn.append(loss_data["Nd_tn"])
            Nd_tt.append(loss_data["Nd_tt"])

            # Accumulate the training time
            T_tn.append(loss_data["t_tn"])

        # Compile the learning data available
        Neps_tot = np.sum(Neps)
        Nd_mean = (np.mean(Nd_tn), np.mean(Nd_tt))
        T_tn_tot = np.sum(T_tn)

        LData = [np.hstack(Loss_tn), np.hstack(Loss_tt)]
        if Eval_tte:
            LData.append(np.hstack(Eval_tte))

        roster_data[student_name] = LData
        Nplot = np.max(Nplot, len(LData))
        xplim = np.max(xplim, Neps_tot)

        # Compute the training time
        student_summary = ru.get_student_summary(
            student_name, Neps_tot, Nd_mean, T_tn_tot, LData, Nln
        )

        learning_summary += student_summary

    # Compile the learning summary footer
    learning_summary += [f"{'=' * Nln}"]

    # Print the learning summary
    console.print(*learning_summary)

    # Create a figure and a set of subplots
    titles = ["Training", "Testing", "TTE"]
    ylabels = ["Loss (log scale)", "Loss (log scale)", "TTE (m)"]
    _, axs = plt.subplots(1, Nplot, figsize=(5, 3))

    # Plot the losses
    for student_name, LData in roster_data.items():
        for idx,ldata in enumerate(LData):
            axs[idx].plot(ldata[0,:],ldata[1,:], label=student_name)
            
    for idx in range(Nplot):
        axs[idx].set_xlim(0, xplim)
        axs[idx].set_yscale('log')
        axs[idx].set_title(titles[idx])
        axs[idx].set_xlabel('Epoch')
        axs[idx].set_ylabel(ylabels[idx])
        axs[idx].legend(loc='upper right')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show(block=False)

def plot_deployments(cohort_name: str, course_name: str, roster: List[str], plot_show: bool = False):
    """
    Plot the simulations for each student in the roster.
    """

    # Initialize the rich variables
    console = ru.get_console()
    table = ru.get_deployment_table()

    # Add Expert to Roster
    roster = ["expert"] + roster

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    deployment_folder = os.path.join(workspace_path, "cohorts", cohort_name, "deployment_data")

    for pilot in roster:
        # Load the deployment data
        try:
            deployment_path = os.path.join(deployment_folder,"sim_"+course_name+"_"+pilot+".pt")
            Trajectories = torch.load(deployment_path)
            
            # Get pilot metrics
            metrics = fh.compute_flight_metrics(Trajectories)
            table = ru.update_deployment_table(table,pilot,metrics)
        except:
            console.print(f"Pilot [bold cyan]{pilot}[/] does not have a deployment.")
            continue

        # Plot the trajectories for each pilot
        if plot_show:
            console.print(f"Plotting trajectories for [bold cyan]{pilot}[/]...")
            p3.RO_to_3D(Trajectories,plot_last=True)

    # Print the summary table
    console.print(table)