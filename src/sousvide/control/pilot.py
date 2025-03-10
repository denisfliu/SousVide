import numpy as np
import torch
import time
import os
import json
import numpy.typing as npt
import albumentations as A

from typing import Dict,Union,Tuple,Literal
from albumentations.pytorch import ToTensorV2
from sousvide.control.policy import Policy

class Pilot():
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
            name:           Name of the pilot.
            hz:             Frequency of the pilot.

            path:           Path to the pilot.
            policy_type:    Type of policy.

            device:         Device to use.
            model:          Nerual network policy.

            process_image:  Image processing function.
            txu_pr:         Previous time/state.
            znn_cr:         Current feature vector.

            tx_cr:          Current state.
            Obj:            Objective.
            Img:            Image.

            hy_idx:         History index.
            tXU:            Time/State/Input history.
            Znn:            Feature vector history.

            da_cfg:         Data augmentation configuration.
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
        obj_dim = [1,18]                                                    # Objective dimensions
        img_dim = [1,3,224,224]                                               # Image dimensions
        
        # Some useful paths
        workspace_path  = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_config_path = os.path.join(
            workspace_path,"configs","pilots",pilot_name+".json")
        pilot_path = os.path.join(
            workspace_path,"cohorts",cohort_name,'roster',pilot_name)
        pilot_config_path = os.path.join(
            pilot_path,"config.json")

        # Check if config file exists, if not create one
        if os.path.isfile(pilot_config_path):
            pass
        else:
            os.makedirs(pilot_path,exist_ok=True)
            with open(default_config_path) as json_file:
                profile = json.load(json_file)

            with open(pilot_config_path, 'w') as outfile:
                json.dump(profile, outfile, indent=4)

        # Load config file
        with open(pilot_config_path) as json_file:
            profile = json.load(json_file)

        # Torch intermediate variables
        use_cuda = torch.cuda.is_available()                                    # Check if cuda is available

        ## Class Variables =================================================================================================
        
        # ---------------------------------------------------------------------
        # Pilot Identifier Variables
        # ---------------------------------------------------------------------

        # Necessary Variables for Base Controller -----------------------------
        self.name = pilot_name
        self.hz = hz

        self.path = pilot_path
        self.policy_type = "None" if profile["type"] is None else profile["type"]

        # ---------------------------------------------------------------------
        # Pilot Neural Network Policy Variables
        # ---------------------------------------------------------------------

        self.device  = torch.device("cuda:0" if use_cuda else "cpu")
        self.model = Policy(profile).to(self.device)
        
        nhy,Nz = int(self.model.nhy),int(self.model.Nz)

        # ---------------------------------------------------------------------
        # Pilot Observe Variables
        # ---------------------------------------------------------------------

        # Function Variables
        self.process_image = process_image                      # Image Processing Function
        self.txu_pr = torch.zeros(1,Ntx+Nu).to(self.device)     # Previous State
        self.znn_cr = torch.zeros(1,Nz).to(self.device)         # Current Feature Vector

        # Network Input Variables
        self.tx_cr = torch.zeros((1,Ntx)).to(self.device)       # Current State
        self.Obj = torch.zeros(obj_dim).to(self.device)         # Objective
        self.Img = torch.zeros(img_dim).to(self.device)         # Image

        # ---------------------------------------------------------------------
        # Pilot Orient Variables
        # ---------------------------------------------------------------------

        # Function Variables
        self.hy_idx = 0

        # Network Input Variables
        self.tXU = torch.zeros((1,nhy,Ntx+Nu)).to(self.device)  # time/State/Input History
        self.Znn = torch.zeros((1,nhy,Nz)).to(self.device)      # Feature Vector History
        
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

    def set_mode(self,mode:Literal['train','deploy']):
        """
        Function that switches the pilot between training mode and deployment.

        Args:
            mode: Mode to be switched to: 'train' or 'deploy'.

        Returns:
            None
        """

        if mode == 'train':
            self.model.train()                              # Set model to training mode
        elif mode == 'deploy':
            self.model.eval()                               # Set model to evaluation mode

            xnn = self.decide()                             # Create dummy information to initialize the model
            self.act(xnn)                                   # Do a 'wipe out' to initialize the model
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'deploy'.")
    
    def observe(self,
                upr:Union[np.ndarray,torch.Tensor],
                tcr:Union[float,torch.Tensor],xcr:Union[np.ndarray,torch.Tensor],
                obj:Union[np.ndarray,torch.Tensor],
                icr:Union[npt.NDArray[np.uint8],None,torch.Tensor],zcr:Union[torch.Tensor,None]) -> None:
        """
        Function that performs the observation step of the OODA loop where we take in all the relevant
        flight data.

        Args:
            upr:    Previous control input.
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            obj:    Objective vector.
            img_cr: Current image frame.
            znn_cr: Current feature vector (None if not available).

        Returns:
            None
        """

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        upr = torch.tensor(upr,dtype=torch.float32).to(self.device).unsqueeze(0)
        tcr = torch.tensor(tcr,dtype=torch.float32).to(self.device).reshape(1,1)
        xcr = torch.from_numpy(xcr).float().to(self.device).unsqueeze(0)
        obj = torch.from_numpy(obj).float().to(self.device).unsqueeze(0)
        zcr = zcr.to(self.device).unsqueeze(0) if zcr is not None else torch.zeros(1,self.Znn.shape[-1]).to(self.device)
        
        # Process image if it is not downsampled
        if icr is None or icr.shape != self.Img.shape:
            icr = self.process_image(icr)
        else:
            icr = torch.from_numpy(icr).float()

        # Update Function Variables
        self.txu_pr.copy_(torch.cat((self.tx_cr,upr),dim=1))
        self.tx_cr.copy_(torch.cat((tcr,xcr),dim=1))
        self.znn_cr.copy_(zcr)

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
        self.tXU[0,self.hy_idx,:] = self.txu_pr     # Update History Input
        self.Znn[0,self.hy_idx,:] = self.znn_cr     # Update History Odometry

        # Increment History Index
        self.hy_idx += 1                            # Increment History index
        if self.hy_idx >= self.tXU.shape[1]:        # If history blocks are full, reset index
            self.hy_idx = 0
        
    def decide(self) -> Dict[str,torch.Tensor]:
        """
        Build the inputs to the neural network model.

        Returns:
            inputs: Dictionary of inputs to the neural network model.
        """

        khy,Nhy = self.hy_idx-1,self.tXU.shape[1]
        idx_hy = (torch.arange(Nhy,0,-1)+khy)%Nhy

        inputs = {
            "rgb_image": self.Img,
            "objective": self.Obj,
            "current": self.tx_cr,
            "history": self.tXU[:,idx_hy,:],
            "feature": self.Znn[:,idx_hy,:]
        }
        
        return inputs
    
    def act(self,xnn: Dict[str,torch.Tensor]) -> Tuple[np.ndarray,torch.Tensor,Union[np.ndarray,None]]:

        """
        Function that performs a forward pass of the neural network model and extracts
        the relevant information.

        Args:
            xnn:    Input to the neural network model.

        Returns:
            unn:    Output from the neural network model.
            znn:    Feature vector output from the feature extractor.
            xnn:    Inputs to the neural network model.
        """

        with torch.no_grad():
            unn,znn,xnn = self.model(xnn)

        # Convert inputs to numpy array
        unn = unn.cpu().numpy().squeeze()

        return unn,znn,xnn

    def OODA(self,
             upr:np.ndarray,
             tcr:float,xcr:np.ndarray,
             obj:np.ndarray,
             icr:Union[npt.NDArray[np.uint8],None],zcr:Union[torch.Tensor,None]) -> Tuple[
                 np.ndarray,
                 Dict[str,torch.Tensor],
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
            zcr:    Current odometry information.

        Returns:
            unn:    Output from the neural network model.
            xnn:    Inputs to the neural network model.
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
        ynn,znn,xnn = self.act(xnn)
        t4 = time.time()

        # Get the total time taken
        tsol = np.array([t1-t0,t2-t1,t3-t2,t4-t3])

        return ynn,znn,xnn,tsol
    
    def control(self,
                upr:np.ndarray,
                tcr:float,xcr:np.ndarray,
                obj:np.ndarray,
                icr:Union[npt.NDArray[np.uint8],None],zcr:Union[torch.Tensor,None]) -> Tuple[
                    np.ndarray,
                    torch.Tensor,
                    Union[np.ndarray,None],
                    np.ndarray]:
        """
        Name mask for the OODA control loop. Variable position swap to match generic controllers.
        
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
        unn,znn,_,tsol = self.OODA(upr,tcr,xcr,obj,icr,zcr)
        
        return unn,znn,tsol