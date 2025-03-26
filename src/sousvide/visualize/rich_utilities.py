import numpy as np

from rich import get_console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from typing import Tuple,List

console = get_console()

def get_generation_progress() -> Progress:
    """
    Create a Rich Progress instance for visualizing generation tasks.
    
    Returns:
        progress:   A Progress object configured for generation tasks.
    """
    
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[bold green3] {task.completed:>2}/{task.total} {task.fields[units]}"),
        TimeRemainingColumn(),
        console=console,
    )
    return progress

def get_training_progress() -> Progress:
    """
    Create a Rich Progress instance for visualizing training tasks.
    
    Returns:
        progress:   A Progress object configured for training tasks.
    """
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description} | [bold dark_green]Loss[/]: [dark_green]{task.fields[loss]:.4f}[/]"),
        BarColumn(),
        TextColumn("[bold green3] {task.completed:>2}/{task.total} {task.fields[units]}"),
        TimeRemainingColumn(),
        console=console,
    )
    return progress

def get_data_description(name:str,value:int, subunits:str=None, Nmn:int=10,Nct:int=10) -> str:
    """"
    Generate a description for the data progress.
    
    Args:
        name:   Name of the data.
        value:  Number of data points."

    Returns:
        data_desc:   Formatted string describing the course and number of data points.
    """


    name_desc = f"[bold green3]{name:<{Nmn}}[/][dark_green]"

    if subunits is not None:
        value_desc = f"({value:>4} {subunits})".ljust(Nct)
    else:
        value_desc = f"{value:>4.5}"

    data_desc = name_desc + value_desc
    
    return data_desc

def get_deployment_table() -> str:
    """
    Generate a table header for deployment data.
    
    Returns:
        table:   Formatted string for the deployment table header.
    """

    # Create a table
    table = Table(title=f"Deployment Summary")

    # Add columns
    table.add_column("Pilot", justify="left")
    table.add_column("TTE Mean", justify="center")
    table.add_column("TTE Best", justify="center")
    table.add_column("PP", justify="center")
    table.add_column("Hz Mean", justify="center")
    table.add_column("Hz Worst", justify="center")

    return table

def update_deployment_table(table:Table,pilot:str,metrics:dict):
    """
    Update the deployment table with metrics data.
    
    Args:
        table:   The deployment table to update.
        pilot:   Name of the pilot.
        metrics:   Dictionary containing the metrics data.
        
    Returns:
        table:   Updated deployment table with metrics.
    """
    
    # Add a row with the metrics
    table.add_row(
        pilot,
        f"{metrics['TTE']['mean']:.2f}",
        f"{metrics['TTE']['best']:.2f}",
        f"{metrics['PP']:.2f}",
        f"{metrics['hz']['mean']:.2f}",
        f"{metrics['hz']['worse']:.2f}"
    )
    
    return table

def get_student_summary(student:str,
                        Neps_tot:int,Nd_mean:Tuple[int],T_tn_tot:int,
                        LData:List[np.ndarray],Nln:int=70) -> str:
    
    # Extract Learning Data
    loss_tn,loss_tt = LData[0][-1,-1],LData[1][-1,-1]
    if len(LData) > 2:
        show_eval = True
        eval_tte = LData[2][-1,-1]
    else:
        show_eval = False

    # Compute the training time
    hours = T_tn_tot // 3600
    minutes = (T_tn_tot % 3600) // 60
    seconds = T_tn_tot % 60

    # Prepare the summary fields
    student_field = f"Student: [bold cyan]{student}[/]".ljust(31)
    tepochs_field = f"Epochs: {Neps_tot}".ljust(13)
    datsize_field = f"Data Size: {Nd_mean[0]}/{Nd_mean[1]}"
    tt_time_field = f"Time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s".ljust(17)
    tn_loss_field = f"[bold green]Train: {loss_tn:.4f}[/]".ljust(13)
    tt_loss_field = f"Test: {loss_tt:.4f}".ljust(12)
    ev_loss_field = f"[bold bright_green]Eval TTE: {eval_tte:.2f}[/]".ljust(15)

    summary = [
            f"{'-' * Nln}\n"
            f"{student_field} | {tepochs_field} | {datsize_field}\n"
    ]

    if show_eval:
        summary.append(
            f"{tt_time_field} | {tn_loss_field} | {tt_loss_field} | {ev_loss_field}\n"
        )
    else:
        summary.append(
            f"{tt_time_field} | {tn_loss_field} | {tt_loss_field}\n"
        )

    return summary