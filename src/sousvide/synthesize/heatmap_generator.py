# Import the relevant modules
import torch
torch.set_float32_matmul_precision('high')

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sousvide.synthesize.rollout_generator as rg
import sousvide.utilities.feature_utilities as fu
import figs.visualize.generate_videos as gv
from scipy.spatial.transform import Rotation as R
import figs.utilities.config_helper as ch
import figs.dynamics.quadcopter_specifications as qs
import figs.utilities.transform_helper as th


def generate_heatmap_data(cohort_name:str,gsplat_name:str,frame_name:str):

    # Load Data
    gsplat = ch.get_gsplat(gsplat_name)
    bframe = ch.get_config(frame_name,'frames')
    bspecs = qs.generate_specifications(bframe)
    camera = gsplat.generate_output_camera(bspecs["camera"])

    Tc2b = bspecs["Tc2b"]
    fx,fy = bframe["camera"]["fx"],bframe["camera"]["fy"]
    cx,cy = bframe["camera"]["cx"],bframe["camera"]["cy"]
    K = np.array([[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]])

    Tb2c = np.linalg.inv(Tc2b)
    
    # Some useful constants
    np.random.seed(1)

    Np = 16
    img_h, img_w = 360,640
    pch_h, pch_w = img_h/Np, img_w/Np
    pch_x = np.linspace(pch_w/2, img_w-pch_w/2, Np)
    pch_y = np.linspace(pch_h/2, img_h-pch_h/2, Np)
    sigma = 30.0

    # Generate the camera poses
    Ndt = 5
    p_tW = np.array([ 0.03, -1.05, -1.20])
    Xdt = generate_Xdt(Ndt, p_tW)

    # Test the camera poses by plotting them
    fig, ax = plt.subplots(Ndt, 1, figsize=(3, 15))

    p_tW_h = np.ones((4,))
    p_tW_h[0:3] = p_tW
    for i in range(Ndt):
        # Get current image
        Tb2w = th.x_to_T(Xdt[i,:])
        Tc2w = Tb2w@Tc2b
        iro,_ = gsplat.render_rgb(camera,Tc2w)

        # Generate the overlay
        Ri = R.from_quat(Xdt[i,6:10]).as_matrix()
        ti = Xdt[i,0:3]

        Tb2w = np.eye(4)
        Tb2w[0:3,0:3] = Ri
        Tb2w[0:3,3] = ti
        Tw2b = np.linalg.inv(Tb2w)

        Tw2c = Tb2c@Tw2b
        p_tc_h = Tw2c@p_tW_h
        p_tc_h[1:3] = -p_tc_h[1:3]  # Flip y and z for camera frame
        p_tc = p_tc_h[0:3]

        uv = K@p_tc[0:3]
        uv = uv/uv[2]
        u,v = uv[0], uv[1]
        u,v = np.clip(u,0,img_w-1), np.clip(v,0,img_h-1)

        heat_ref = torch.zeros((Np,Np))
        for j in range(Np):
            for k in range(Np):
                heat_ref[j,k] = np.exp(-((pch_x[j]-u)**2 + (pch_y[k]-v)**2)/(2*sigma**2))

        overlay = fu.heatmap_overlay(heat_ref,iro)
        ax[i].imshow(overlay)

        def generate_Xdt(Ndt:int,p_tW:np.ndarray):
        # Some useful constants
        z_gW = np.array([0.0, 0.0, 1.0])

        # Generate the dataset
        Xdt = np.zeros((Ndt, 10))
        for i in range(Ndt):
            # Sample spherical coordinates
            x_cW = np.random.uniform(-1.0, 1.0)
            y_cW = np.random.uniform(-1.0, 1.0)
            z_cW = np.random.uniform(-0.2,-1.5)

            p_cW = np.array([x_cW, y_cW, z_cW])

            xb = p_tW-p_cW
            xb = xb/np.linalg.norm(xb)
            yb = np.cross(z_gW,xb)
            zb = np.cross(xb, yb)
            R_cW = np.array([xb, yb, zb]).T

            Xdt[i,0:3] = p_cW
            Xdt[i,6:10] = R.from_matrix(R_cW).as_quat()
        
        return Xdt