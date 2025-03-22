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

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(5, 3))

    # Plot the losses for each student
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
        Loss_tn, Loss_tt, Eval_tte, Neps = [], [], [], 0
        Nd_tn, Nd_tt = [], []
        T_tn = 0
        for loss_data in losses.values():
            # Add the loss data to the lists
            Loss_tn.append(loss_data["Loss_tn"])
            Loss_tt.append(loss_data["Loss_tt"])

            # Add the evaluation data to the list, adjusting for the number of episodes
            if loss_data["Eval_tte"] is not None:
                Eval_tte.append(np.array(loss_data["Eval_tte"]))

            # Update the total number of episodes and other metrics
            Neps += loss_data["N_eps"]

            # Append the number of data points for training and testing
            Nd_tn.append(loss_data["Nd_tn"])
            Nd_tt.append(loss_data["Nd_tt"])

            # Accumulate the training time
            T_tn += loss_data["t_tn"]

        Loss_tn = np.hstack(Loss_tn)
        Loss_tt = np.hstack(Loss_tt)
        Eval_tte = np.vstack(Eval_tte)

        # Compute the training time
        student_summary = ru.get_student_summary(
            student_name, Neps, Nd_tn, Nd_tt,
            Loss_tn[-1], Loss_tt[-1], Eval_tte[-1,-1], T_tn, Nln
        )

        learning_summary += student_summary

        # Plot the losses
        axs[0].plot(Loss_tn, label=student_name)
        axs[1].plot(Loss_tt, label=student_name)

        axs[2].plot(Eval_tte[:,0],Eval_tte[:,1], label=student_name)
        axs[2].set_xlim(0, Neps)

        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')

    # Compile the learning summary footer
    learning_summary += [f"{'=' * Nln}"]

    # Print the learning summary
    console.print(*learning_summary)

    axs[0].set_title('Training')
    axs[0].legend(loc='upper right')

    axs[1].set_title('Testing')
    axs[1].legend(loc='upper right')

    axs[2].set_title('TTE')
    axs[2].legend(loc='upper right')

    # Set common labels
    for ax in axs[0:2]:
        ax.set(xlabel='Epoch', ylabel='Loss (log scale)')
    axs[2].set(xlabel='Epoch', ylabel='TTE (m)')

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