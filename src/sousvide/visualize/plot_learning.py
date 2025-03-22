import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import sousvide.visualize.plot_3D as p3
import sousvide.visualize.rich_utilities as ru
import sousvide.flight.flight_helper as fh

from typing import List

def plot_losses(cohort_name:str, roster:List[str], network_name:str,Nln:int=65):
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
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

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
        Loss_tn, Loss_tt, Neps = [], [], 0
        Nd_tn, Nd_tt = [], []
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
        seconds = np.around(T_tn % 60, 1)

        # Compile the student summary
        student_summary = [
            f"{'-' * Nln}\n"
            f"Student: [bold cyan]{student_name.center(10)}[/bold cyan] | "
            f"Total Epochs: {Neps} | "
            f"Data Size: {Nd_tn[-1]}/{Nd_tt[-1]}\n"
            f"[bold green]Train Loss: {np.around(Loss_tn[-1], 4)}[/]  | Test Loss: {np.around(Loss_tt[-1], 4)}  | "
            f"Time: {hours}h {minutes}m {seconds}s"
        ]

        learning_summary += student_summary

        # Plot the losses
        axs[0].plot(Loss_tn, label=student_name)
        axs[1].plot(Loss_tt, label=student_name)

        axs[0].set_yscale('log')
        axs[1].set_yscale('log')

    # Compile the learning summary footer
    learning_summary += [f"{'=' * Nln}"]

    # Print the learning summary
    console.print(*learning_summary)

    axs[0].set_title('Training Loss')
    axs[0].legend(loc='upper right')

    axs[1].set_title('Testing Loss')
    axs[1].legend(loc='upper right')

    # Set common labels
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Loss (log scale)')

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