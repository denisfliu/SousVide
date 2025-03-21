import numpy as np
import torch
import os
import sousvide.visualize.plot_3D as p3
import sousvide.visualize.plot_time as pt
import sousvide.visualize.rich_utilities as ru

from rich.text import Text

from typing import List

def plot_rollout_data(cohort:str,Nsamples:int=50,
                      show_3D:bool=True,show_time:bool=True):
    """"
    Plot the rollout data for a cohort.
    """

    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    rollout_path = os.path.join(cohort_path,"rollout_data")

    # Initialize the console variable
    console = ru.get_console()
    subunits = "dpts"

    # Load Flight Data
    Ntot = 0
    courses_desc = []
    for course in os.listdir(rollout_path):
        course_path = os.path.join(rollout_path, course)
        
        # Extract all files in child folder "trajectories"
        traj_folder_path = os.path.join(course_path, "trajectories")
        traj_files = os.listdir(traj_folder_path)
        traj_file_paths = [os.path.join(traj_folder_path, f) for f in traj_files]

        # Extract diagnostics data
        Ndsets = len(traj_file_paths)
        Ntot += Ndsets

        Ndata = 0
        for traj_file_path in traj_file_paths:
            trajectories = torch.load(traj_file_path)

            for trajectory in trajectories:
                Ndata += trajectory["Ndata"]

        course_desc = ru.get_data_description(course, Ndata, subunits=subunits) + f" [{Ndsets} datasets]"
        courses_desc.append(course_desc)
    
        # Plot example trajectory
        if show_3D == True or show_time == True:
            # Load a random trajectory file
            traj_file = np.random.choice(traj_files)
            traj_file_path = os.path.join(traj_folder_path, traj_file)
            trajectories = torch.load(traj_file_path)
            
            # Trim the number of samples
            if Nsamples > len(trajectories):
                Nsamples = len(trajectories)
                console.print(f"Only {Nsamples} samples available in {traj_file}. Showing all samples.")
            else:
                console.print(f"Showing {Nsamples} samples from {traj_file}")
            trajectories = trajectories[0:Nsamples]

        # Plot the data
        if show_3D == True:
            p3.RO_to_3D(trajectories,scale=0.5)
        if show_time == True:
            pt.RO_to_time(trajectories)

    # Collate diagnostics
    courses_desc = [Text.from_markup(course_desc).plain for course_desc in courses_desc]  # Strip rich tags
    courses_desc = "\n".join(courses_desc)  # Join descriptions into a single string
    console.print(
        f"Rollout produced {Ntot} datasets with the following courses: \n"
        f"{courses_desc}", style="white")
        

# def plot_observation_data(cohort:str,roster:List[str],random:bool=True):
#     """"
#     Plot the observation data for a cohort.
#     """

#     # Generate some useful paths
#     workspace_path = os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#     cohort_path = os.path.join(workspace_path,"cohorts",cohort)
#     observation_data_path = os.path.join(cohort_path,"observation_data")

#     print("==========================================================================================")
#     print("Observation Data Summary")
#     print("==========================================================================================")

#     print("Cohort Name :",cohort)

#     # Review the observation data
#     for pilot_name in roster:
#         # Get a sample topic path in the pilot's directory
#         pilot_path = os.path.join(observation_data_path,pilot_name)
#         topic_path = next(os.scandir(pilot_path)).path

#         observation_files = []
#         for root, _, files in os.walk(topic_path):
#             for file in files:
#                 observation_files.append(os.path.join(root, file))

#         Nobsf = len(observation_files)

#         # Get some data insights
#         observation_files = sorted(observation_files)

#         # Get approximate number of observations
#         observations = torch.load(observation_files[0])
#         Ntrain = max(1,(Nobsf-1))*observations["Ndata"]         # Training and Test is the same set when Nobsf = 1
#         Ntest = observations["Ndata"]

#         # Load pilot
#         pilot = Pilot(cohort,pilot_name)

#         print("------------------------------------------------------------------------------------------")
#         print(f"Pilot Name              : {pilot.name}")
#         print(f"Neural Network(s)       : {list(pilot.policy.networks.keys())}")
#         print(f"Approx. Train/Test Ratio: {Ntrain}/{Ntest}")

#     print("==========================================================================================")
