import numpy as np
import torch
import os
import sousvide.visualize.plot_3D as p3
import sousvide.visualize.plot_time as pt
import sousvide.visualize.rich_utilities as ru

from typing import List

def plot_rollout_data(cohort:str,Nsamples:int=50,random:bool=True):
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

    # Load Flight Data
    for course in os.listdir(rollout_path):
        course_path = os.path.join(rollout_path, course)
        
        # Extract all files in child folder "trajectories"
        trajs_path = os.path.join(course_path, "trajectories")
        trajs_files = os.listdir(trajs_path)
                
        # Load a .pt file
        if random == True:
            trajs_file = np.random.choice(trajs_files)
        else:
            trajs_file = trajs_files[0]

        trajs_file_path = os.path.join(trajs_path, trajs_file)
        trajectories = torch.load(trajs_file_path)

        # Trim the number of samples
        if Nsamples > len(trajectories):
            Nsamples = len(trajectories)
            console.print(f"Only {Nsamples} samples available in {trajs_file}. Showing all samples.")
        else:
            console.print(f"Showing {Nsamples} samples from {trajs_file}")
        trajectories = trajectories[0:Nsamples]

        # Plot the data
        p3.RO_to_3D(trajectories,scale=0.5)
        pt.RO_to_time(trajectories)

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
