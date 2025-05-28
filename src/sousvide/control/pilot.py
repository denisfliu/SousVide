import numpy as np
import torch
import time
import os
import json
import numpy.typing as npt
import albumentations as A
import figs.utilities.config_helper as ch

from typing import Literal
from figs.control.base_controller import BaseController
from albumentations.pytorch import ToTensorV2
from sousvide.control.policy import Policy

class Pilot(BaseController):
    def __init__(self,cohort_name:str,pilot_name:str,
                 hz:int=20,Ntxu:int=15,Nft:int=6):
        """
        Initializes a pilot object. 
        
        Args:
            cohort_name:    Name of the cohort.
            pilot_name:     Name of the pilot.
            hz:             Frequency of the pilot.
            Ntxu:           Number of time/state/input variables.
            Nft:            Number of force/torque variables.
        
        Variables:
            path:           Path to the pilot's directory.
            device:         Device to run the pilot on (CPU or GPU).
            policy:         Policy object for the pilot.
            process_image:  Function to process the image.
            txu_cr:         Current time, state and input variables.
            rgb_cr:         Current RGB image.
            fts_cr:         Current force/torque sensor data.
            pch_cr:         Current feature map.
            sqc_idx:        Current index in the sequence.
            Nsqc:           Number of history blocks.
            Sqc:            Sequence arrays containing EKF, CoinFT and feature map data.
            txu_pr:         Previous time, state and input variables.
            fts_pr:         Previous force/torque sensor data.
            pch_pr:         Previous feature map.
            da_cfg:         Data augmentation configuration.
            name:           Name of the pilot.
            hz:             Frequency of the pilot.
        """
        
        ## Initialization =================================================================================================
        
        # Initialize paths
        workspace_path  = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        pilot_path = os.path.join(
            workspace_path,"cohorts",cohort_name,'roster',pilot_name)
        
        if not os.path.exists(pilot_path):
            os.makedirs(pilot_path, exist_ok=True)
            
        # Initialize pytorch variables
        profile = ch.get_config(pilot_name,"pilots")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy = Policy(profile,pilot_name,pilot_path).to(device)

        # Initialize sequence variables
        Nsqc = policy.Nhy
        Sqc = {
            "ekf": torch.zeros((1,Nsqc,Ntxu)).to(device),          # EKF data (dt, p, v, q, u)
            "cft": torch.zeros((1,Nsqc,Nft)).to(device),           # CoinFT data
            "map": torch.zeros((1,Nsqc,16,16,768)).to(device)      # Feature map data
        }

        # Initialize image processing variables
        transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])            
        process_image = lambda x: transform(image=x)["image"]
        img_dim = [1,3,224,224]

        ## Class Variables =================================================================================================
        
        # Path Variables
        self.path = pilot_path

        # Neural Network Variables
        self.device = device
        self.policy = policy

        # Image Processing Variables
        self.process_image = process_image

        # Policy Input Variables
        self.txu_cr = torch.zeros((1,Ntxu)).to(device)         # Current Time,State and Input
        self.rgb_cr = torch.zeros(img_dim).to(device)          # Current Image
        self.fts_cr = torch.zeros((1,6)).to(device)            # Current Force/Torque sensor data

        # Feature Map Variables
        self.pch_cr = torch.zeros((1,16,16,768)).to(device)    # Current Feature Map

        # Sequence Variables
        self.sqc_idx = 0                                        # Current index in the sequence
        self.Nsqc = Nsqc                                        # Number of history blocks  
        self.Sqc = Sqc                                          # Sequence arrays

        self.txu_pr = torch.zeros((1,Ntxu)).to(device)         # Previous Time,State and Input
        self.fts_pr = torch.zeros((1,6)).to(device)            # Previous Force/Torque sensor data
        self.pch_pr = torch.zeros((1,16,16,768)).to(device)    # Previous Feature Map

        # Data Augmentation
        self.da_cfg = {
            "type": profile["data_augmentation"]["type"],
            "mean": profile["data_augmentation"]["mean"],
            "std": profile["data_augmentation"]["std"]
        }

        # Necessary Variables for Base Controller -----------------------------
        self.name = pilot_name
        self.hz = hz

    def set_mode(self,mode:Literal['train','deploy']):
        """
        Function that switches the pilot between training mode and deployment.

        Args:
            mode: Mode to be switched to: 'train' or 'deploy'.

        Returns:
            None
        """

        if mode == 'train':
            self.policy.train()         # Set model to training mode
        elif mode == 'deploy':
            self.policy.eval()          # Set model to evaluation mode

            xnn = self.collate()        # Create dummy information to initialize the model
            self.act(xnn)               # Do a 'wipe out' to initialize the model
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'deploy'.")

    def reset_memory(self,x0:np.ndarray,u0:np.ndarray=None,
                     fts0:np.ndarray=None,pch0:torch.Tensor=None) -> None:
        """
        Function that sets the initial states of the pilot.

        Args:
            x0: Initial state.
            u0: Initial control input.
            fts0: Initial force/torque sensor data.
            pch0: Initial feature map.

        Returns:
            None
        """

        # Fill u0,ft0 and pch0 if they are None
        if u0 is None:
            u0 = np.array([-0.4,0.0,0.0,0.0])

        if fts0 is None:
            fts0 = np.zeros(6)

        if pch0 is None:
            pch0 = torch.zeros((16,16,768)).to(self.device)

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        x0 = torch.from_numpy(x0).float().to(self.device).unsqueeze(0)
        u0 = torch.from_numpy(u0).float().to(self.device).unsqueeze(0)
        fts0 = torch.from_numpy(fts0).float().to(self.device).unsqueeze(0)
        pch0 = pch0.unsqueeze(0).to(self.device)  # Ensure pch0 is a 4D tensor
        
        # Set the initial sequence states
        self.txu_pr[0,1:11] = self.txu_cr[0,1:11] = x0
        self.txu_pr[0,11:15] = self.txu_cr[0,11:15] = u0
        self.fts_pr = self.fts_cr = fts0
        self.pch_pr = self.pch_cr = pch0

        for i in range(self.Sqc["ekf"].shape[1]):
            self.Sqc["ekf"][0,i,0] = 0.0
            self.Sqc["ekf"][0,i,1:11] = x0
            self.Sqc["ekf"][0,i,11:15] = u0

            self.Sqc["cft"][0,i,:] = fts0
            self.Sqc["map"][0,i,:,:,:] = pch0

    def observe(self,t_cr:float,x_cr:np.ndarray,u_pr:np.ndarray,
                rgb_cr:np.ndarray|None,dpt_cr:None,
                fts_cr:np.ndarray) -> None:
        """
        Function that intakes the observable data from the environment and updates the
        pilot's internal variables.

        Args:
            t_cr:   Current flight time.
            x_cr:   Current state in world frame (observed).
            u_pr:   Previous control input.
            rgb_cr: Current image frame.
            dpt_cr: Current depth frame.
            fts_cr: Current force/torque sensor data.

        Returns:
            None
        """

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        t_cr = torch.tensor(t_cr,dtype=torch.float32).to(self.device).reshape(1,1)
        x_cr = torch.from_numpy(x_cr).float().to(self.device).unsqueeze(0)
        u_pr = torch.tensor(u_pr,dtype=torch.float32).to(self.device).unsqueeze(0)

        # Process image if it is not downsampled
        if rgb_cr is None or rgb_cr.shape != self.rgb_cr.shape:
            rgb_cr = self.process_image(rgb_cr)
        else:
            rgb_cr = torch.from_numpy(rgb_cr).float()
        # Depth image is not used in the pilot, so we ignore it
        _ = dpt_cr

        # Convert force/torque sensor data to torch tensor
        fts_cr = torch.from_numpy(fts_cr).float().to(self.device).unsqueeze(0)

        # Update previous variables
        self.txu_pr[0,0:11],self.txu_pr[0,11:15] = self.txu_cr[0,0:11],u_pr
        self.fts_pr[0,:] = self.fts_cr[0,:]
        self.pch_pr[0,:,:,:] = self.pch_cr[0,:,:,:]

        # Update current variables
        self.txu_cr[0,0],self.txu_cr[0,1:11] = t_cr,x_cr        
        self.rgb_cr[0,:,:,:] = rgb_cr
        self.fts_cr = fts_cr

    def retain(self):
        """
        Function that retains observable data within the sequence variables.

        Args:
            None

        Returns:
            None
        """

        # Update sensor based sequence variables
        dt = self.txu_cr[0,0]-self.txu_pr[0,0]
        xu = self.txu_pr[0,1:15]

        self.Sqc["ekf"][0,self.sqc_idx,:] = torch.hstack((dt,xu))
        self.Sqc["cft"][0,self.sqc_idx,:] = self.fts_pr[0,:]

        # Update the feature map sequence variable
        self.Sqc["map"][0,self.sqc_idx,:,:,:] = self.pch_pr[0,:,:,:]

        # Increment sequence index
        self.sqc_idx += 1                   # Increment History index
        if self.sqc_idx >= self.Nsqc:       # If history blocks are full, reset index
            self.sqc_idx = 0
        
    def collate(self) -> dict[str,torch.Tensor]:
        """
        Collates the inputs to the neural network model.

        Returns:
            Xnn: Dictionary of inputs to the neural network model.
        """

        # Determine the current sequence index and the number of history blocks
        k_sqc,N_sqc = self.sqc_idx-1,self.Nsqc
        idx_sqc = (torch.arange(N_sqc,0,-1)+k_sqc)%N_sqc
        
        # Extract the relevant data
        xnn_tx,xnn_rgb,xnn_fts = self.txu_cr[:,0:11],self.rgb_cr,self.fts_cr
        xnn_ekf = self.Sqc["ekf"][:,idx_sqc,:]
        xnn_cft = self.Sqc["cft"][:,idx_sqc,:]
        xnn_map = self.Sqc["map"][:,idx_sqc,:,:,:]

        # Collate into a list of inputs to the neural network model
        Xnn = {
            "current": xnn_tx, "rgb_image": xnn_rgb, "wrench": xnn_fts,
            "dynamics": xnn_ekf, "wrenches": xnn_cft, "feature_arrays": xnn_map
        }
        
        return Xnn
    
    def act(self,Xnn:dict[str,torch.Tensor]) -> tuple[np.ndarray,torch.Tensor,dict[str,list[torch.Tensor]]]:

        """
        Function that performs a forward pass of the neural network model and extracts
        the relevant information.

        Args:
            Xnn:    Input dictionary into the neural network model.

        Returns:
            unn:    Output from the neural network model.
            Dnn:    Input/Label data for training the neural network models.
        """

        # Query the neural network model
        with torch.no_grad():
            unn,znn,Dnn = self.policy(Xnn)

        # Update class variables
        if unn is not None:
            self.txu_cr[0,11:15] = unn              # Update the current control input
        if znn is not None:
            self.pch_cr[0,:,:,:] = znn              # Store the feature map in the pilot
        
        # Post-process the output
        unn = unn.cpu().numpy().squeeze()       # Convert command to numpy array
        
        return unn,Dnn

    def ORCA(self,t_cr:float,x_cr:np.ndarray,u_pr:np.ndarray,
                rgb_cr:np.ndarray,dpt_cr:np.ndarray,fts_cr:np.ndarray) -> tuple[
                 np.ndarray,
                 dict[str,torch.Tensor],
                 dict[str,list[torch.Tensor]],
                 np.ndarray]:
        """
        Function that runs the ORCA loop. This is the main function that is called by the
        pilot during flight.

        Args:
            t_cr:   Current flight time.
            x_cr:   Current state in world frame (observed).
            u_pr:   Previous control input.
            rgb_cr: Current RGB image data.
            dpt_cr: Current depth image data.
            fts_cr: Current force/torque sensor data.

        Returns:
            unn:    Output from the neural network model.
            Xnn:    Inputs/Outputs to the neural networks.
            tsol:   Time taken to solve components of the OODA loop in list form.
        """
        
        # Get the current time
        t0 = time.time()

        # Perform the OODA loop
        self.observe(t_cr,x_cr,u_pr,rgb_cr,dpt_cr,fts_cr)
        t1 = time.time()
        self.retain()
        t2 = time.time()
        Xnn = self.collate()
        t3 = time.time()
        unn,Dnn = self.act(Xnn)
        t4 = time.time()

        # Get the total time taken
        tsol = np.array([t1-t0,t2-t1,t3-t2,t4-t3])

        return unn,Dnn,tsol
    
    def control(self,t_cr:float,x_cr:np.ndarray,u_pr:np.ndarray,
                rgb_cr:np.ndarray,dpt_cr:np.ndarray,fts_cr:np.ndarray
    ) -> tuple[np.ndarray,dict[str,torch.Tensor],np.ndarray]:
        """
        Name mask for the OODA control loop.
        
        Args:
            t_cr:   Current flight time.
            x_cr:   Current state in world frame (observed).
            u_pr:   Previous control input.
            rgb_cr: Current RGB image data.
            dpt_cr: Current depth image data.
            fts_cr: Current force/torque sensor data.

        Returns:
            unn:    Control input.
            Aux:    Auxiliary outputs (solve time).
        """
        unn,_,tsol = self.ORCA(t_cr,x_cr,u_pr,rgb_cr,dpt_cr,fts_cr)

        # Compute auxiliary outputs
        Aux = {"tsol":tsol}  # Auxiliary outputs

        return unn,Aux