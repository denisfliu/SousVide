"""
Helper functions for sous vide.
"""

import numpy as np
import figs.utilities.transform_helper as th
from scipy.spatial.transform import Rotation as R

def compute_prms(frame:dict[str,np.ndarray,str|int|float]) -> list:
    """
    Computes the frame parameters (mass, thrust coefficient, normalized thrust gain).

    Args:
        frame:  Frame configuration.

    Returns:
        params: Frame parameters (mass, thrust coefficient, normalized thrust gain).
    """

    # Unpack variables
    n_mtr = frame["number_of_rotors"]
    m_fr = frame["mass"]
    k_fr = frame["motor_thrust_coeff"]

    # Compute mass normalized thrust
    c_fr = (k_fr*n_mtr)/m_fr

    # Params
    params = [m_fr,k_fr,c_fr]

    return params

def compute_Fres(Xro:np.ndarray,Uro:np.ndarray,Fro:np.ndarray,
             frame:dict[str,np.ndarray,str|int|float],
             bframe:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the resultant forces acting on the frame.

    Args:
        Xro:    State vector.
        Uro:    Control input vector.
        Fro:    External forces.
        frame:  Frame configuration.
        bframe: Base frame configuration.

    Returns:
        Fres:   Resultant forces array.
    """

    # Some useful constants
    g = np.array([0,0,9.81])        # Gravity vector
    zb = np.array([0.0,0.0,1.0])    # Z-axis unit vector

    # Unpack variables
    Ndt = Uro.shape[0]
    n_mtr = frame["number_of_rotors"]
    m_fr,m_bs = frame["mass"],bframe["mass"]
    k_fr,k_bs = frame["motor_thrust_coeff"],bframe["motor_thrust_coeff"]
    
    # Compute the resultant forces
    Fres = np.zeros((Ndt,3))
    for i in range(Ndt):
        # Unpack data
        xcr = Xro[i,:]
        ucr = Uro[i,:]
        fcr = Fro[i,:]

        # Compute rotation matrix
        Rb2w = R.from_quat(xcr[6:10]).as_matrix()

        # Compute forces
        f_dgv = (m_fr-m_bs)*g                   # Difference from gravity
        f_dth = (k_fr-k_bs)*n_mtr*ucr[0]*zb     # Difference from thrust

        Fres[i,:] = f_dgv + Rb2w@f_dth + fcr

    return Fres

def compute_FOro(Tro:np.ndarray,Xro:np.ndarray,Uro:np.ndarray,
               Fro:np.ndarray,frame:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the flat output sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        Uro:    Control input vector.
        Fro:    Force vector.
        frame:  Frame configuration.
    
    Returns:
        FO:    Flat output rollout sequence.
        
    """

    # Unpack variables
    tXU = np.hstack((Tro[:-1].reshape((-1,1)),Xro[:-1,:],Uro))

    # Compute the flat output
    _,FO = th.tXU_to_TsFO(tXU,Fro,frame)

    return FO