import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sousvide.synthesize.observation_generator as og

from torch.utils.data import DataLoader
from tqdm.notebook import trange
from sousvide.control.pilot import Pilot
from sousvide.control.policy import Policy
from sousvide.control.networks.base_net import BaseNet
from sousvide.instruct.synthesized_data import *
from typing import List
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# Initialize Console
console = Console()

def train_roster(cohort_name:str,roster:List[str],
                 network_name:str,Neps:int,regen_data:bool=False,
                 lim_sv:int=10,lr:float=1e-4,batch_size:int=64):
    
    # Regenerate observation data
    if regen_data:
        console.print("Regenerating observation data...")
        og.generate_observation_data(cohort_name,roster)
    else:
        console.print("Using existing observation data...")

    for student_name in roster:
        # Train the student
        train_student(cohort_name,student_name,network_name,Neps,lim_sv,lr,batch_size)

def train_student(cohort_name:str,student_name:str,
                  network_name:str,Neps:int,
                  lim_sv:int=10,lr:float=1e-4,batch_size:int=64):

    # Pytorch Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    criterion = nn.MSELoss(reduction='mean')

    # Load the student
    student = Pilot(cohort_name,student_name)

    # Extract network if it exists
    if network_name not in student.policy.networks:
        console.print(f"Network {network_name} not found in student {student_name}.")
        return
    else:
        network:BaseNet = get_network(student.policy,network_name)    

    # Set parameters to optimize over
    opt = optim.Adam(network.parameters(),lr=lr)

    # Some Useful Paths
    student_path = student.path
    losses_path  = os.path.join(student_path,"losses_"+network_name+".pt")
    network_path = os.path.join(student_path,network_name+".pt")

    # Load loss log if it exists
    if os.path.exists(losses_path):
        prev_losses_log = torch.load(losses_path)
    else:
        prev_losses_log = {}

    # Initialize the loss entry
    loss_entry = {
        "network": network_name,
        "N_eps": None, "Nd_tn": None, "Nd_tt": None, "t_tn": None,
        "Loss_tn": [], "Loss_tt": []
    }

    # Record the start time
    start_time = time.time()

    # Training + Testing Loop
    Loss_tn,Loss_tt = [],[]
    with Progress(
        TextColumn("[bold cyan]Maverick > {network_name}[/bold cyan] | Epoch {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("Loss: [bold yellow]{task.fields[loss]:.4f}[/bold yellow]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Initialize PTask
        epoch_task = progress.add_task("Training...", total=Neps, loss=0.0)

        for ep in range(Neps):
            # Get Observation Data Files (Paths)
            od_train_files,od_test_files = get_data_paths(cohort_name,student.name,network_name)

            # Training
            epLosses_tn,Ndata_tn = [],0
            for od_train_file in od_train_files:
                # Load Datasets
                dataset = generate_dataset(od_train_file,device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

                # Training
                for input,label in dataloader:
                    # Forward Pass
                    prediction = network(*input)
                    loss = criterion(prediction,label)

                    # Backward Pass
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    # Save loss logs
                    epLosses_tn.append(label.shape[0]*loss.item())
                    Ndata_tn += label.shape[0]

            # Testing
            epLosses_tt,Ndata_tt = [],0
            for od_test_file in od_test_files:
                # Load Datasets
                dataset = generate_dataset(od_test_file,device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

                # Testing
                for input,label in dataloader:
                    # Forward Pass
                    prediction = network(*input)
                    loss = criterion(prediction,label)

                    # Save loss logs
                    epLosses_tt.append(label.shape[0]*loss.item())
                    Ndata_tt += label.shape[0]

            # Loss Diagnostics
            epLoss_tn = sum(epLosses_tn)/Ndata_tn
            epLoss_tt = sum(epLosses_tt)/Ndata_tt
            Loss_tn.append(epLoss_tn)
            Loss_tt.append(epLoss_tt)
            
            # Update Progress
            progress.update(epoch_task,advance=1,loss=epLoss_tn)

            # Save at intermediate steps and at the end
            if ((ep+1) % lim_sv == 0) or (ep+1==Neps):
                # Record the end time
                end_time = time.time()
                t_train = end_time - start_time

                torch.save(network,network_path)

                loss_entry["Loss_tn"] = Loss_tn
                loss_entry["Loss_tt"] = Loss_tt
                loss_entry["N_eps"] = ep+1
                loss_entry["Nd_tn"] = Ndata_tn
                loss_entry["Nd_tt"] = Ndata_tt
                loss_entry["t_tn"] = t_train

                # Save Loss
                timestamp = time.strftime("%y%m%d_%H%M%S")
                log_name = f"log_{timestamp}"

                curr_losses_log = prev_losses_log.copy()
                curr_losses_log[log_name] = loss_entry
                
                torch.save(curr_losses_log,losses_path)

        # Cap off progress update
        progress.refresh()

def get_network(policy:Policy,net_name:Union[str,List[str]]) -> BaseNet:
    """
    Get the networks based on the list of network names.

    Args:
        policy:     The policy object.
        net_name:   The list of network names or a single network name.

    Returns:
        network:    The target network.
    """

    # Get the network
    network = policy.networks[net_name]
    
    # Switch from network forward pass to label pass (if it exists)
    network.use_fpass = False

    # Lock/Unlock the network
    network.train()
    for param in network.parameters():
        param.requires_grad = True
    
    return network