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
                 hz:int=20,Ntx:int=11,Nu:int=4):
        """
        Initializes a pilot object. 
        
        Args:
            cohort_name:    Name of the cohort.
            pilot_name:     Name of the pilot.
            hz:             Frequency of the pilot.
            Ntx:            Number of time/state variables.
            Nu:             Number of control variables.
        
        Variables:
        """

        ## Initial Variables ===============================================================================================

        # Some useful constants
        transform = A.Compose([                                             # Image transformation pipeline
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])            
        process_image = lambda x: transform(image=x)["image"]               # Image processing
        img_dim = [1,3,224,224]                                               # Image dimensions
        
        # Generate pilot path
        workspace_path  = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        pilot_path = os.path.join(
            workspace_path,"cohorts",cohort_name,'roster',pilot_name)

        # Create the pilot path if it does not exist        
        if not os.path.exists(pilot_path):
            os.makedirs(pilot_path)

        # Load the pilot configuration
        profile = ch.get_config(pilot_name,"pilots")
        
        # Torch intermediate variables
        use_cuda = torch.cuda.is_available()                                    # Check if cuda is available

        ## Class Variables =================================================================================================
        
        # ---------------------------------------------------------------------
        # Pilot Neural Network Policy Variables
        # ---------------------------------------------------------------------

        self.path = pilot_path
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.policy = Policy(profile,pilot_name,pilot_path).to(self.device)

        # ---------------------------------------------------------------------
        # Pilot Observe Variables
        # ---------------------------------------------------------------------
        
        # Function Variables
        self.process_image = process_image                      # Image Processing Function
        self.txupr = torch.zeros(1,Ntx+Nu).to(self.device)      # Previous Time,State and Input
        self.zcr = zcr                                          # Current Feature Vector

        # Network Input Variables
        self.txcr = torch.zeros((1,Ntx)).to(self.device)       # Current State
        self.Img = torch.zeros(img_dim).to(self.device)         # Image

        # ---------------------------------------------------------------------
        # Pilot Orient Variables
        # ---------------------------------------------------------------------

        # Function Variables
        self.hy_idx = 0

        # Network Input Variables
        self.Dnn = torch.zeros((1,nhy,18)).to(self.device)              # time step/State/Input History
        self.Znn = Znn
        
        # ---------------------------------------------------------------------
        # Pilot Training Variables
        # ---------------------------------------------------------------------
                
        # Data Augmentation
        self.da_cfg = {
            "type": profile["data_augmentation"]["type"],
            "mean": profile["data_augmentation"]["mean"],
            "std": profile["data_augmentation"]["std"]
        }
        
        # ---------------------------------------------------------------------
        # Pilot Identifier Variables
        # ---------------------------------------------------------------------

        # Necessary Variables for Base Controller -----------------------------
        self.name = pilot_name
        self.hz = hz
        self.nhy = nhy

        # ---------------------------------------------------------------------

    def set_mode(self,mode:Literal['train','deploy']):
        """
        Function that switches the pilot between training mode and deployment.

        Args:
            mode: Mode to be switched to: 'train' or 'deploy'.

        Returns:
            None
        """

        if mode == 'train':
            self.policy.train()                              # Set model to training mode
        elif mode == 'deploy':
            self.policy.eval()                               # Set model to evaluation mode

            xnn = self.decide()                             # Create dummy information to initialize the model
            self.act(xnn)                                   # Do a 'wipe out' to initialize the model
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'deploy'.")
    
    def generate_feature_variables(self,nhy:int) -> tuple[dict[str,torch.Tensor],dict[str,torch.Tensor]]:
        """
        Function that generates znn and Znn variables for the neural network model.
        """

        znn,Znn = {},{}
        for key,network in self.policy.networks.items():
            _,fp_size,_ = network.get_io_sizes()

            znn[key] = torch.zeros((1,fp_size))
            Znn[key] = torch.zeros((1,nhy,fp_size))

        return znn,Znn

    def set_initial_memory(self,x0:np.ndarray,u0:np.ndarray|None=None) -> None:
        """
        Function that sets the initial states of the pilot.

        Args:
            x0: Initial state.
            u0: Initial control input.

        Returns:
            None
        """

        # Educated guess for hover
        if u0 is None:
            u0 = np.array([-0.4,0.0,0.0,0.0])        

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        x0 = torch.from_numpy(x0).float().to(self.device).unsqueeze(0)
        u0 = torch.from_numpy(u0).float().to(self.device).unsqueeze(0)
        
        # Set the initial states
        self.txupr[0,1:11] = self.txcr[0,1:11] = x0
        self.txupr[0,11:15] = u0

        for i in range(self.Dnn.shape[1]):
            self.Dnn[0,i,1:11] = x0
            self.Dnn[0,i,11:15] = u0

    def observe(self,
                upr:np.ndarray|torch.Tensor,
                tcr:float|torch.Tensor,xcr:np.ndarray|torch.Tensor,
                obj:np.ndarray|torch.Tensor,
                icr:npt.NDArray[np.uint8]|None|torch.Tensor,zcr:dict[str,torch.Tensor]) -> None:
        """
        Function that performs the observation step of the OODA loop where we take in all the relevant
        flight data.

        Args:
            upr:    Previous control input.
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            obj:    Objective vector.
            icr:    Current image frame.
            zcr:    Dictionary of current feature vectors.

        Returns:
            None
        """

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        upr = torch.tensor(upr,dtype=torch.float32).to(self.device).unsqueeze(0)
        tcr = torch.tensor(tcr,dtype=torch.float32).to(self.device).reshape(1,1)
        xcr = torch.from_numpy(xcr).float().to(self.device).unsqueeze(0)
        obj = torch.from_numpy(obj).float().to(self.device).unsqueeze(0)

        for value in zcr.values():
            value = value.to(self.device).unsqueeze(0)

        # Process image if it is not downsampled
        if icr is None or icr.shape != self.Img.shape:
            icr = self.process_image(icr)
        else:
            icr = torch.from_numpy(icr).float()

        # Update Function Variables
        self.txupr.copy_(torch.cat((self.txcr,upr),dim=1))
        self.txcr.copy_(torch.cat((tcr,xcr),dim=1))

        for key,value in zcr.items():
            self.zcr[key].copy_(value)

        # Update Network Input Variables
        self.Obj.copy_(obj)
        self.Img.copy_(icr)

    def orient(self):
        """
        Function that performs the orientation step of the OODA loop where we generate the history
        variables from the flight data.

        Args:
            None

        Returns:
            None
        """

        # Update history data
        dt0 = self.txcr[0,0]-self.txupr[0,0]
        p0,q0 = self.txupr[0,1:4],self.txupr[0,7:11]
        v0,v1 = self.txupr[0,4:7],self.txcr[0,4:7]
        a0,u0 = (v1-v0)/(dt0+1e-9),self.txupr[0,11:15]

        self.Dnn[0,self.hy_idx,:] = torch.hstack((dt0,p0,v0,a0,q0,u0))

        # Update Feature Vector History
        for network in self.zcr:
            self.Znn[network][0,self.hy_idx,:] = self.zcr[network]

        # Increment History Index
        self.hy_idx += 1                                                # Increment History index
        if self.hy_idx >= self.Dnn.shape[1]:                           # If history blocks are full, reset index
            self.hy_idx = 0
        
    def decide(self) -> dict[str,torch.Tensor]:
        """
        Build the inputs to the neural network model.

        Returns:
            inputs: List of inputs to the neural network model.
        """

        khy,Nhy = self.hy_idx-1,self.Dnn.shape[1]
        idx_hy = (torch.arange(Nhy,0,-1)+khy)%Nhy
        
        xnn_im,xnn_ob = self.Img, self.Obj
        xnn_cr = self.txcr
        xnn_hy = self.Dnn[:,idx_hy,:]
        xnn_ft = {key: self.Znn[key][:,idx_hy,:] for key in self.Znn.keys()} 


        # Generate the inputs to the neural network model
        inputs = [xnn_im,xnn_ob,xnn_cr,xnn_hy,xnn_ft]
        
        return inputs
    
    def act(self,
            xnn: list[torch.Tensor]) -> tuple[
                np.ndarray,
                torch.Tensor,
                dict[str,list[torch.Tensor]]]:

        """
        Function that performs a forward pass of the neural network model and extracts
        the relevant information.

        Args:
            xnn:    Input to the neural network model.

        Returns:
            unn:    Output from the neural network model.
            znn:    Feature vector output from the feature extractor.
            Xnn:    Syllabus inputs to the neural networks.
        """

        with torch.no_grad():
            unn,znn,Xnn = self.policy(*xnn)

        # Post-process the outputs
        unn = unn.cpu().numpy().squeeze()       # Convert command to numpy array
        
        for value in znn.values():
            value = value.squeeze()                     # Flatten the feature vector

        return unn,znn,Xnn

    def OODA(self,
             upr: np.ndarray,
             tcr: float,
             xcr: np.ndarray,
             obj: np.ndarray,
             icr: npt.NDArray[np.uint8]|None,
             zcr: dict[str,torch.Tensor]) -> tuple[
                 np.ndarray,
                 dict[str,torch.Tensor],
                 dict[str,list[torch.Tensor]],
                 np.ndarray]:
        """
        Function that runs the OODA loop. This is the main function that is called by the
        pilot during flight.

        Args:
            upr:    Previous control input.
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            obj:    Objective vector.
            icr:    Current image frame.
            zcr:    Dictionary of current feature vectors.

        Returns:
            unn:    Output from the neural network model.
            znn:    Feature vector output.
            Xnn:    Inputs/Outputs to the neural networks.
            tsol:   Time taken to solve components of the OODA loop in list form.
        """
        
        # Get the current time
        t0 = time.time()

        # Perform the OODA loop
        self.observe(upr,tcr,xcr,obj,icr,zcr)
        t1 = time.time()
        self.orient()
        t2 = time.time()
        xnn = self.decide()
        t3 = time.time()
        unn,znn,Xnn = self.act(xnn)
        t4 = time.time()

        # Get the total time taken
        tsol = np.array([t1-t0,t2-t1,t3-t2,t4-t3])

        return unn,znn,Xnn,tsol
    
    def control(
        self,Xcr:dict[str,np.ndarray]
    ) -> tuple[np.ndarray,dict[str,torch.Tensor],np.ndarray]:
        """
        Name mask for the OODA control loop.
        
        Args:
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            upr:    Previous control input.
            obj:    Objective vector.
            icr:    Current image frame.
            zcr:    Input feature vector.
        
        Returns:
            unn:    Control input.
            tsol:   Time taken to solve components of the OODA loop in list form.
        """
        unn,znn,_,tsol = self.OODA(Xcr)
        
        return unn,znn,tsol