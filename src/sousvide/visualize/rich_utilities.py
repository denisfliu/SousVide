from rich import get_console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

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
        TimeElapsedColumn(),
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
        TimeElapsedColumn(),
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
