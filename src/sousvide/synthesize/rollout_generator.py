import numpy as np
import os
import copy
import torch
import figs.utilities.trajectory_helper as th
import figs.tsplines.min_snap as ms
import sousvide.synthesize.synthesize_helper as sh
import sousvide.visualize.rich_utilities as ru

from typing import Dict,Union,Tuple,List
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC

def generate_rollout_data(cohort_name:str,course_names:List[str],
                          scene_name:str,method_name:str,
                          expert_name:str="vrmpc_fr",bframe_name:str="carl",
                          Nro_ds:int=50,use_compress:bool=False) -> None:
    
    """
    Generates rollout data for a given cohort. The rollout data comprises a set of courses
    flown within a given scene on variations of a specific drone frame by a specific pilot
    using a user-defined data generation method. The rollout data is saved to as pairs of
    .pt files, one for trajectory data and one for image data, in a directory corresponding
    to a combination of the course, scene and method names.

    Args:
        cohort_name:    Directory to store the rollout data (and later the roster of pilots).
        course_names:   List of trajectory courses to be flown.
        scene_name:     3D reconstruction of the scene contained as a Gaussian Splat.
        method_name:    Data generation method detailing the sampling and simulation configs.
        expert_name:    Expert pilot (default is a vrmpc) to fly the trajectories.
        bframe_name:    Base frame for flying the trajectories (default is carl).
        Nro_sv:         Number of rollouts per save.
        use_compress:   Compress the image data.

    Returns:
        None:           (flight data saved to cohort directory)
    """

    # Initialize the progress variables
    progress = ru.get_generation_progress()
    subunits = "dpts"
    sample_desc1 = "[bold dark_green]Generating rollouts...[/]"
    sample_desc2 = "[bold dark_green]Saving dataset...[/]"

    # Load configs
    method_config = sh.load_config(method_name,"method")
    expert_config = sh.load_config(expert_name,"pilots")
    bframe_config = sh.load_config(bframe_name,"frame")

    # Initialize the simulator
    simulator = Simulator(scene_name,method_config["rollout"])

    # Generate rollouts for each course
    with progress:
        # Initialize sample progress bar
        sample_task = progress.add_task(sample_desc1,total=None,units='samples')

        for course_name in course_names:
            # Load and name the course_config
            course_config = sh.load_config(course_name,"course")

            # Compute desired trajectory
            output = ms.solve(course_config)
            if output is not False:
                Tpd,CPd = output
            else:
                raise ValueError("Desired trajectory not feasible. Aborting.")
            
            # Get the batches of sample start times
            Tsp_bts = sh.compute_Tsp_batches(
                Tpd[0], Tpd[-1], method_config["duration"],
                method_config["rate"], method_config["reps"], Nro_ds
            )

            # Initialize course progress bar
            Ndata = 0
            course_desc = ru.get_data_description(course_name,Ndata,subunits=subunits)
            course_task = progress.add_task(course_desc,
                total=len(Tsp_bts), units='datasets')

            # Generate Sample Set Batches
            for idx_bt,Tsp_bt in enumerate(Tsp_bts):
                # Generate sample frames and perturbations
                Frames = generate_frames(
                    Tsp_bt, bframe_config, method_config["randomization"]["parameters"]
                )
                Perturbations = generate_perturbations(
                    Tsp_bt, Tpd, CPd, method_config["randomization"]["initial"]
                )

                # Update the samples progress bar config
                progress.reset(sample_task,description=sample_desc1,total=len(Frames))
                sample_bar = (progress,sample_task)

                # Generate rollout data
                Trajectories,Images = generate_rollouts(
                    simulator,course_config,expert_config,
                    Frames,Perturbations,
                    method_config["duration"],method_config["tol_select"],
                    idx_bt,sample_bar)

                # Update the observations progress bar
                progress.update(sample_task,description=sample_desc2)

                # Save the rollout data
                save_rollouts(cohort_name,course_name,
                            Trajectories,Images,
                            idx_bt,use_compress)

                # Update the data count
                Ndata += sum([trajectory["Ndata"] for trajectory in Trajectories])

                # Update the progress bar
                course_desc = ru.get_data_description(course_name,Ndata,subunits=subunits)
                progress.update(course_task,
                                description=course_desc,advance=1)

            # Ensure progress catches last update
            progress.refresh()

def generate_frames(Tsps:np.ndarray,
                    base_frame_config:Dict[str,Union[int,float,List[float]]],
                    parameters_bounds:List[float],
                    rng_seed:Union[int,None]=None) -> List[Dict[str,Union[np.ndarray,str,int,float]]]:
    
    """
    Generates a list of drone variations for a given base drone configuration. The configurations are
    generated by perturbing the base drone configuration with bounded uniform noise. The number of
    configurations generated is determined by the sample set config dictionary.

    Args:
        Tsps:               Sample times.
        base_frame_config:  Base frame configuration dictionary.
        parameters_bounds:  Frame sample set config dictionary.
        rng_seed:           Random number generator seed.

    Returns:
        Drones:             List of drone configurations (dictionary format).
    """

    # TODO: Generalize this to nnio config
    parameters_key = ["mass","force_normalized"]

    # Sample Count
    Nsps = len(Tsps)

    # Set random number generator seed
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Generate Drone Frames
    Frames = []
    for _ in range(Nsps):
        # Instantiate a new frame
        frame = copy.deepcopy(base_frame_config)

        # Randomize the frame
        for idx,key in enumerate(parameters_key):
            bounds = [-parameters_bounds[idx],parameters_bounds[idx]]
            frame[key] += np.random.uniform(*bounds)

        # Save to a dictionary
        Frames.append(frame)

    return Frames

def generate_perturbations(Tsps:np.ndarray,
                           Tpd:np.ndarray,CPd:np.ndarray,
                           initial_bounds:List[float],
                           rng_seed:int=None) -> List[Dict[str,Union[float,np.ndarray]]]:
    """
    Generates a list of perturbed initial states for the drone given an ideal trajectory. The perturbed
    initial states are generated by sampling a random initial times and corresponding state vectors from
    the ideal trajectory using a bounded uniform distribution. The state vectors are then perturbed with
    uniform noise. The number of perturbed initial states generated is determined by the sample set
    config dictionary.

    Args:
        Tsps:                   Sample times.
        Tpd:                    Ideal trajectory times.
        CPd:                    Ideal trajectory control points.
        initial_bounds:         Initial state bounds.
        rng_seed:               Random number generator seed.

    Returns:
        Perturbations:          List of perturbations (dictionary format).
    """

    # Sample Count
    Nsps = len(Tsps)

    # Set random number generator seed
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    # Unpack the config
    w_x0 = np.array(initial_bounds,dtype=float)

    # Get ideal trajectory for quaternion checking
    tXUd = th.TS_to_tXU(Tpd,CPd,None,10)

    # Generate perturbed starting points    
    Perturbations = []
    for i in range(Nsps):
        # Sample random start time and get corresponding state vector sample
        t0 = Tsps[i]
        idx0 = np.where(Tpd <= t0)[0][-1]
        idx0 = min(idx0,(len(Tpd)-2))
        t00,t0f = Tpd[idx0],Tpd[idx0 + 1]

        x0s = th.ts_to_xu(t0-t00,t0f-t00,CPd[idx0,:,:],None)
        
        # Perturb state vector sample
        w0 = np.random.uniform(-w_x0,w_x0)
        x0 = x0s + w0
        
        # Ensure quaternion is well-behaved (magnitude and closest to previous)
        idxr = np.where(tXUd[0,:] <= t0)[0][-1]
        x0[6:10] = th.obedient_quaternion(x0[6:10],tXUd[7:11,idxr])

        # Store perturbation in list
        perturbation = {"t0":t0,"x0":x0}
        Perturbations.append(perturbation)
    
    return Perturbations

def generate_rollouts(
        sim:Simulator,
        course_config:Dict[str,Union[np.ndarray,List[np.ndarray]]],
        policy_config:Dict[str,Union[int,float,List[float]]],
        Frames:Dict[str,Union[np.ndarray,str,int,float]],
        Perturbations:Dict[str,Union[float,np.ndarray]],
        Tdt_ro:float,tol_select:float,
        idx_set:int,
        progress_bar:Tuple[ru.Progress,int]=None
        ) -> Tuple[List[Dict[str,Union[np.ndarray,np.ndarray,np.ndarray]]],List[torch.Tensor]]:
    """
    Generates rollout data for the quadcopter given a list of drones and initial states (perturbations).
    The rollout comprises trajectory data and image data. The trajectory data is generated by running
    the MPC controller on the quadcopter for a fixed number of steps. The trajectory data consists of
    time, states [p,v,q], body rate inputs [fn,w], objective state, data count, solver timings, advisor
    data, rollout id, and course name. The image data is generated by rendering the quadcopter at each
    state in the trajectory data. The image data consists of the image data and the data count.

    Args:
        simulator:      Simulator object.
        course_config:  Course configuration dictionary.
        policy_config:  Policy configuration dictionary.
        Frames:         List of drone configurations.
        Perturbations:  List of perturbed initial states.
        Tdt_ro:         Rollout duration.
        tol_select:     Error tolerance.
        idx_set:        Index of the rollout set.
        progress_bar:   Progress bar (if available).

    Returns:
        Trajectories:   List of trajectory rollouts.
        Images:         List of image rollouts.
    """
    
    # Get console
    console = ru.get_console()

    # Unpack the trajectory
    Tpi,CPi = ms.solve(course_config)
    obj = sh.ts_to_obj(Tpi,CPi)
    tXd = th.TS_to_tXU(Tpi,CPi,None,10)
    
    # Initialize rollout variables
    Trajectories,Images = [],[]
    ctl = VehicleRateMPC(course_config,policy_config)

    # Set the tolerance if undefined
    if tol_select is None:
        tol_select = np.inf

    # Rollout the trajectories
    Ndata = len(Perturbations)
    for idx,(frame_config,perturbation) in enumerate(zip(Frames,Perturbations)):
        # Unpack rollout variables
        t0,x0 = perturbation["t0"],perturbation["x0"]
        dt = Tdt_ro or (Tpi[-1] - t0)
        tf = t0 + dt

        # Load the simulation variables
        sim.load_frame(frame_config)
        ctl.update_frame(frame_config)    

        # Simulate the flight
        Tro,Xro,Uro,Iro,Fro,Tsol = sim.simulate(ctl,t0,tf,x0)

        # Check if the rollout data is useful
        err = np.min(np.linalg.norm(tXd[1:4,:]-Xro[0:3,-1].reshape(-1,1),axis=0))
        if err < tol_select:
            # Package the rollout data
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,"Fro":Fro,
                "tXd":tXd,"obj":obj,"Ndata":Uro.shape[1],"Tsol":Tsol,
                "rollout_id":str(idx_set).zfill(3)+str(idx).zfill(3),
                "frame":frame_config}

            images = {
                "images":Iro,
                "rollout_id":str(idx_set).zfill(3)+str(idx).zfill(3)
            }

            # Store rollout data
            Trajectories.append(trajectory)
            Images.append(images)

            # Update the progress bar
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,advance=1)
        else:
            # console.print(
            #     f"[bold red]Rollout failed to meet tolerance. Skipping...[/]\n"
            #     f"Euclidean Distance: {err:.3f} > {tol_select:.3f}")
            
            Ndata -= 1
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,total=Ndata)
            
    return Trajectories,Images

def save_rollouts(cohort_name:str,course_name:str,
                  Trajectories:List[Tuple[np.ndarray,np.ndarray,np.ndarray]],
                  Images:List[torch.Tensor],
                  stack_id:Union[str,int],
                  use_compression:bool=False) -> None:
    """
    Saves the rollout data to a .pt file in folders corresponding to coursename within the cohort 
    directory. The rollout data is stored as a list of rollout dictionaries of size stack_size for
    ease of comprehension and loading (at a cost of storage space).
    
    Args:
        cohort_path:    Cohort path.
        course_name:    Name of the course.
        Trajectories:   Rollout data.
        Images:         Image data.
        stack_id:       Stack id.
        use_compression: Compress the image data.

    Returns:
        None:           (rollout data saved to cohort directory)
    """
    # Create rollout course directory (if it does not exist)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)
    
    course_path = os.path.join(cohort_path,"rollout_data",course_name)
    traj_course_path = os.path.join(course_path,"trajectories")
    imgs_course_path = os.path.join(course_path,"images")

    if not os.path.exists(traj_course_path):
        os.makedirs(traj_course_path)
    
    if not os.path.exists(imgs_course_path):
        os.makedirs(imgs_course_path)

    # Save the stacks
    dset_name = str(stack_id).zfill(3) if type(stack_id) == int else str(stack_id)
    traj_path = os.path.join(traj_course_path,"trajectories"+dset_name+".pt")
    imgs_path = os.path.join(imgs_course_path,"images"+dset_name+".pt")

    if use_compression:
        Images = sh.compress_data(Images)

    torch.save(Trajectories,traj_path)
    torch.save(Images,imgs_path)

