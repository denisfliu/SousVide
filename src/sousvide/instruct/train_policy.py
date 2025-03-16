import numpy as np
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from tqdm.notebook import trange
from sousvide.control.pilot import Pilot
from sousvide.control.policy import Policy
from sousvide.instruct.synthesized_data import *
from typing import List,Tuple,Literal
from enum import Enum

def train_roster(cohort_name:str,roster:List[str],
                 syllabus:Dict[str,Dict[str,Union[List[str],int]]],
                 lim_sv:int,
                 lr:float=1e-4,batch_size:int=64):
    
    print("==========================================================================")
    for student_name in roster:
        print("Training Student: ",student_name)
        for topic_id,topic in syllabus.items():
            status = train_student(cohort_name,student_name,topic_id,topic,lim_sv,lr,batch_size)
            if not status:
                print(f"{topic_id} does not apply to {student_name}")
            else:
                print(f"{topic_id} completed by {student_name}")
        print("--------------------------------------------------------------------------")
    print("==========================================================================")

def train_student(cohort_name:str,student_name:str,
                  topic_id:str,topic:Dict[str,Union[List[str],str,int]],
                  lim_sv:int,lr:float,batch_size:int):

    # Pytorch Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    criterion = nn.MSELoss(reduction='mean')

    # Load the student
    student = Pilot(cohort_name,student_name)

    # Extract variables using syllabus
    net_active = get_networks(student.policy,topic["active"],False)
    nets_update = get_networks(student.policy,topic["update"],True)    
    Neps = topic["Neps"]

    # Exit if syllabus does not apply to student
    if (net_active is None) or (nets_update is None):
        return False
    
    # Set parameters to optimize over
    opt = optim.Adam(nets_update.parameters(),lr=lr)

    # Some Useful Paths
    student_path = student.path
    losses_path  = os.path.join(student_path,"losses_"+topic_id+".pt")
    
    # Load loss log if it exists
    if os.path.exists(losses_path):
        prev_losses_log = torch.load(losses_path)
    else:
        prev_losses_log = {}

    # Initialize the loss entry
    loss_entry = {
        "active": topic["active"], "update": topic["update"],
        "N_eps": None, "Nd_tn": None, "Nd_tt": None, "t_tn": None,
        "Loss_tn": [], "Loss_tt": []
    }

    # Record the start time
    start_time = time.time()

    # Training + Testing Loop
    Loss_tn,Loss_tt = [],[]
    with trange(Neps) as eps:
        for ep in eps:
            # Get Observation Data Files (Paths)
            od_train_files,od_test_files = get_data_paths(cohort_name,student.name,topic["active"])

            # Training
            epLosses_tn,Ndata_tn = [],0
            for od_train_file in od_train_files:
                # Load Datasets
                dataset = generate_dataset(od_train_file,device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

                # Training
                for input,label in dataloader:
                    # Forward Pass
                    prediction = net_active(*input)
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
                    prediction = net_active(*input)
                    loss = criterion(prediction,label)

                    # Save loss logs
                    epLosses_tt.append(label.shape[0]*loss.item())
                    Ndata_tt += label.shape[0]

            # Loss Diagnostics
            epLoss_tn = sum(epLosses_tn)/Ndata_tn
            epLoss_tt = sum(epLosses_tt)/Ndata_tt
            Loss_tn.append(epLoss_tn)
            Loss_tt.append(epLoss_tt)
            
            eps.set_description('Training Loss %f' % epLoss_tn)

            # Save at intermediate steps and at the end
            if ((ep+1) % lim_sv == 0) or (ep+1==Neps):
                # Record the end time
                end_time = time.time()
                t_train = end_time - start_time

                for net_name,network in nets_update.items():
                    network_path = os.path.join(student_path,net_name+".pt")
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
    
    return True

def get_networks(policy:Policy,net_names:Union[str,List[str]],is_update:bool):
    """
    Get the networks based on the list of network names.

    Args:
        policy:     The policy object.
        net_names:  The list of network names or a single network name.
        is_update:  If true, the networks are updated.

    Returns:
        networks:   The list of networks or a single network.
    """

    # Ensure net_names is a list
    single_net = False
    if isinstance(net_names, str):
        net_names = [net_names]
        single_net = True

    # Extract the networks
    networks = nn.ModuleDict()
    for net_name in net_names:
        # Catch the all case
        if net_name == "all":
            network = policy
        elif net_name in policy.networks:
            network = policy.networks[net_name]
        else:
            return None
        
        # Switch from network forward pass to label pass (if it exists)
        network.use_fpass = False

        # Lock/Unlock the networks
        if is_update:
            network.train()
            for param in network.parameters():
                param.requires_grad = True
        else:
            network.eval()
            for param in network.parameters():
                param.requires_grad = False

        # Add the network to the list
        networks[net_name] = network

    if single_net:
        return networks[net_names[0]]
    
    return networks