import numpy as np
import torch
import os

import figs.utilities.trajectory_helper as th
import figs.visualize.generate_videos as gv

import sousvide.synthesize.synthesize_helper as sh
import sousvide.utilities.sousvide_utilities as svu
import sousvide.visualize.record_flight as rf
import sousvide.visualize.rich_utilities as ru
import sousvide.flight.flight_helper as fh

from typing import List,Literal,Union
from sousvide.control.pilot import Pilot
from figs.tsplines import min_snap as ms
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC

def deploy_roster(cohort_name:str,course_name:str,scene_name:str,method_name:str,
                  roster:List[str],expert_name:str="vrmpc_fr",bframe_name:str="carl",
                  mode:Literal["evaluate","visualize","generate"]="eval",show_table:bool=False) -> Union[None,dict]:
    """"
    Simulate a roster of pilots on a given course within a given scene on
    variations of a specific drone frame using a specified method. This is
    a close mirror to generate_rollout_data with a few key differences; it
    computes flight performance metrics (Trajectory Tracking Error [TTE] 
    and Proximity Percentile [PP]) across multiple rollouts and it produces
    video output for the last trajectory for each pilot. 
    
    Args:
        cohort_name:    Directory to store the rollout data (and later the roster of pilots).
        course_name:    Trajectory course to be flown.
        scene_name:     3D reconstruction of the scene contained as a Gaussian Splat.
        method_name:    Data generation method detailing the sampling and simulation configs.
        roster:         List of pilot names to simulate.
        bframe_name:    Base frame for flying the trajectories (default is carl).
        mode:           Mode of operation for the simulation, can be "evaluate", "visualize", or "generate".
        show_table:     Boolean to print the summary table of flight metrics.

    Returns:
        None:          The function saves the simulation data and video to disk.
    """

    # Extract configs
    method_config = svu.load_config(method_name,"method")
    bframe_config = svu.load_config(bframe_name,"frame")
    course_config = svu.load_config(course_name,"course")

    # Some usefule intermediate variables
    crew = ["expert"]+roster              # Add expert to the roster
    
    # Initialize the simulator
    sim = Simulator(scene_name,method_config["rollout"])

    # Compute desired trajectory spline
    output = ms.solve(course_config)
    if output is not False:
        Tpd,CPd = output
    else:
        raise ValueError("Desired trajectory not feasible. Aborting.")

    # Get the sample start times and end times
    Tsp_bts = sh.compute_Tsp_batches(
        Tpd[0], Tpd[-1], method_config["duration"],
        method_config["rate"], method_config["reps"]
    )
    
    # Generate sample frames and perturbations
    Frames = sh.generate_frames(
        Tsp_bts[0], bframe_config, method_config["randomization"]["parameters"]
    )
    Perturbations = sh.generate_perturbations(
        Tsp_bts[0], Tpd, CPd, method_config["randomization"]["initial"]
    )

    # Initialize the rich variables
    console = ru.get_console()
    table = ru.get_deployment_table()

    # Simulate samples across roster
    Metrics = {}
    for pilot in crew:
        # Load Pilot
        if pilot == "expert":
            ctl = VehicleRateMPC(course_config,expert_name)
        else:
            ctl = Pilot(cohort_name,pilot)
            ctl.set_mode('deploy')

        # Compute ideal trajectory variables
        obj = svu.ts_to_obj(Tpd,CPd)
        tXd = th.TS_to_tXU(Tpd,CPd,None,10)

        # Simulate trajectory across samples
        trajectories = []
        for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):
            # Unpack rollout variables
            t0,x0 = perturbation["t0"],perturbation["x0"]
            dt = method_config["duration"] or (Tpd[-1]-Tpd[0])
            tf = t0+dt

            # Load Frame
            sim.load_frame(frame)

            # Simulate Trajectory
            Tro,Xro,Uro,Iro,Fro,Tsol = sim.simulate(ctl,t0,tf,x0,obj)

            # Save Trajectory
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,"Fro":Fro,
                "tXd":tXd,"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,
                "rollout_id":"sim"+str(0).zfill(3)+str(idx).zfill(3),
                "frame":frame}
            trajectories.append(trajectory)

        # Compile deployment data
        deployment_data = {
            "trajectories":trajectories,
            "video": {"hz":ctl.hz,"images":Iro}
        }

        # Update the metrics table
        metrics = fh.compute_flight_metrics(trajectories)
        table = ru.update_deployment_table(table,pilot,metrics)
        
        # Update the metrics dictionary
        Metrics[pilot] = metrics

        # Save last trajectory as video/flight recorder
        if mode == "visualize":
            save_deployments(cohort_name,course_name,pilot,deployment_data,
                             is_generate=False)
        elif mode == "generate":
            save_deployments(cohort_name,course_name,pilot,deployment_data,
                             is_generate=True)
        elif mode == "evaluate":
            pass  # No action needed for evaluate mode
            
    # Print the summary table
    if show_table:
        console.print(table)
    
    if mode == "evaluate":
        return Metrics
    else:
        return None

def save_deployments(cohort_name:str,course_name:str,pilot_name:str,deployment_data:dict,
                     is_generate:bool=False) -> None:
    
    # Some useful path(s)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    deployment_path = os.path.join(workspace_path,"cohorts",cohort_name,"deployment_data")
    
    # Create the deployment directory if it doesn't exist
    if not os.path.exists(deployment_path):
        os.makedirs(deployment_path)

    # Save the Data
    if is_generate:
        for trajectory in deployment_data["Trajectories"]:
            Tro:np.ndarray = trajectory["Tro"]
            Xro:np.ndarray = trajectory["Xro"]
            Uro:np.ndarray = trajectory["Uro"]
            Tsol:np.ndarray = trajectory["Tsol"]
            tXd,obj = trajectory["tXd"],trajectory["obj"]
            Adv = None

            flight_record = rf.FlightRecorder(
                Xro.shape[0],Uro.shape[0],
                20,tXd[0,-1],[360,640,3],obj,cohort_name,course_name,pilot_name)
            flight_record.simulation_import(images,Tro,Xro,Uro,tXd,Tsol,Adv)
            flight_record.save()        
    else:
        data_name = "sim_"+course_name+"_"+pilot_name
        trajectories = deployment_data["Trajectories"]
        images = deployment_data["video"]["images"]
        hz = deployment_data["video"]["hz"]
        
        trajectories_path = os.path.join(deployment_path,data_name+".pt")
        video_path = os.path.join(deployment_path,data_name+".mp4")

        torch.save(trajectories,trajectories_path)
        gv.images_to_mp4(images,video_path+'.mp4', hz)
