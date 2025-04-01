import numpy as np
import torch
import os

import figs.utilities.config_helper as ch
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
from figs.dynamics.external_forces import ExternalForces

def deploy_roster(cohort_name:str,course_name:str,gsplat_name:str,method_name:str,
                  roster:List[str],expert_name:str="vrmpc_fr",bframe_name:str="carl",
                  mode:Literal["evaluate","visualize","generate"]="evaluate",show_table:bool=False) -> Union[None,dict]:
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
        gsplat_name:    3D reconstruction of the scene contained as a Gaussian Splat.
        method_name:    Data generation method detailing the sampling and simulation configs.
        roster:         List of pilot names to simulate.
        bframe_name:    Base frame for flying the trajectories (default is carl).
        mode:           Mode of operation for the simulation, can be "evaluate", "visualize", or "generate".
        show_table:     Boolean to print the summary table of flight metrics.

    Returns:
        None:          The function saves the simulation data and video to disk.
    """

    # Extract configs
    course = ch.get_config(course_name,"courses")
    gsplat = ch.get_gsplat(gsplat_name)
    method = ch.get_config(method_name,"methods")
    expert = ch.get_config(expert_name,"pilots")
    bframe = ch.get_config(bframe_name,"frames")
    
    # Initialize the simulator
    simulator = Simulator(gsplat,method["rollout"])

    # Compute the desired variables
    Tsd,FOd = ms.solve(course["waypoints"])["FO"]
    Fex = ExternalForces(course["forces"])

    tXUd = th.TsFO_to_tXU(Tsd,FOd,bframe["mass"],bframe["motor_thrust_coeff"],Fex)
    obj = svu.tXU_to_obj(tXUd)

    # Get the batch of sample start times
    t0,tf = Tsd[0],Tsd[-1]
    dt_ro = method["duration"] or tf-t0
    rate = method["rate"] or 1/dt_ro
    reps = method["reps"] or 1

    Tsp_bt = sh.compute_Tsp_batches(t0,tf,dt_ro,rate,reps)[0]

    # Generate sample frames and perturbations
    Frames = sh.generate_frames(
        Tsp_bt, bframe, method["randomization"]["parameters"]
    )
    Perturbations = sh.generate_perturbations(
        Tsp_bt, tXUd, method["randomization"]["initial"]
    )

    # Initialize the rich variables
    console = ru.get_console()
    table = ru.get_deployment_table()


    # Simulate samples across expert+roster
    crew = ["expert"]+roster              # Add expert to the roster

    Metrics = {}
    for pilot in crew:
        # Load Pilot
        if pilot == "expert":
            controller = VehicleRateMPC(expert,course)
        else:
            controller = Pilot(cohort_name,pilot)
            controller.set_mode('deploy')

        # Simulate trajectory across samples
        trajectories = []
        for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):
            # Unpack rollout variables
            t0,x0 = perturbation["t0"],perturbation["x0"]
            tf = t0 + dt_ro

            # Update the simulation variables
            simulator.update_frame(frame)

            # Simulate Trajectory
            Tro,Xro,Uro,Iro,Fro,Tsol = simulator.simulate(controller,t0,tf,x0,obj)

            # Save Trajectory
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,"Fro":Fro,
                "tXUd":tXUd,"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,
                "rollout_id":"sim"+str(0).zfill(3)+str(idx).zfill(3),
                "frame":frame}
            trajectories.append(trajectory)

        # Compile deployment data
        deployment_data = {
            "trajectories":trajectories,
            "video": {"hz":controller.hz,"images":Iro}
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
            tXUd,obj = trajectory["tXUd"],trajectory["obj"]
            Adv = None

            hz = int(1/(Tro[1]-Tro[0]))
            flight_record = rf.FlightRecorder(
                Xro.shape[0],Uro.shape[0],
                hz,tXUd[0,-1],[360,640,3],obj,cohort_name,course_name,pilot_name)
            flight_record.simulation_import(images,Tro,Xro,Uro,tXUd,Tsol,Adv)
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
