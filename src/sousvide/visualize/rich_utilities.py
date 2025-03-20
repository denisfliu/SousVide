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
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[bold green3] {task.completed:>2}/{task.total} {task.fields[units]}"),
        TimeElapsedColumn(),
        console=console,
    )
    return progress

def get_sample_description() -> str:
    """
    Generate a description for the sample progress.
    
    Returns:
        sample_desc:   Formatted string describing the dataset progress.
    """
    
    sample_desc = "[bold dark_green]Dataset Progress...[/]"

    return sample_desc

def get_course_description(course_name:str,Ndata:int) -> str:
    """"
    Generate a description for the course progress.
    
    Args:
        course_name:   Name of the course.
        Ndata:         Number of data points in the course."

    Returns:
        course_desc:   Formatted string describing the course and number of data points.
    """

    main_desc = f"[bold green3]{course_name:<10}[/][dark_green]"
    count_desc = f"({Ndata:>4} dpts)".ljust(10)

    course_desc = main_desc + count_desc
    
    return course_desc
