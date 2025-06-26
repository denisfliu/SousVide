# Import the relevant modules
import torch
torch.set_float32_matmul_precision('high')

import os
import numpy as np
import albumentations as A
import figs.utilities.config_helper as ch
import figs.dynamics.quadcopter_specifications as qs
import figs.utilities.transform_helper as th
import sousvide.utilities.feature_utilities as fu
import sousvide.control.networks.feature_extractors as fe

from scipy.spatial.transform import Rotation as R
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# Initialize image processing variables
transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])            
process_image = lambda x: transform(image=x)["image"]

def generate_circus_data(gsplat_name:str,frame_name:str,p_tW:np.ndarray,Ntot:int,folder:str,
                          Ndps:int=100,Np:int=16,sigma:float=30.0) -> dict[str,np.ndarray]:

    # Some useful constants
    # np.random.seed(10)
    Nds = (Ntot + Ndps - 1) // Ndps

    # Make data directory if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load configs
    gsplat = ch.get_gsplat(gsplat_name)
    bframe = ch.get_config(frame_name,'frames')
    bspecs = qs.generate_specifications(bframe)
    camera = gsplat.generate_output_camera(bspecs["camera"])

    Tc2b = bspecs["Tc2b"]
    Rgl2cv = bspecs["Rgl2cv"]
    img_h, img_w = bspecs["camera"]["height"], bspecs["camera"]["width"]
    K = bspecs["K"]

    Tb2c = np.linalg.inv(Tc2b)

    pch_h, pch_w = img_h/Np, img_w/Np
    pch_u = np.linspace(pch_w/2, img_w-pch_w/2, Np)         # along width
    pch_v = np.linspace(pch_h/2, img_h-pch_h/2, Np)         # along height

    # Load ViT
    vit = fe.DINOv2()

    for idx in range(Nds):
        # Determine the number of data points for this dataset
        Ndt = min(Ntot - idx*Ndps, Ndps)

        # Generate the camera poses
        XbW,PtC,UVc = generate_poses(Ndt,p_tW,K,Tb2c,Rgl2cv,img_h,img_w)

        # Generate image data
        Img,Hmp = torch.zeros((Ndt,3,224,224)), np.zeros((Ndt,img_h,img_w,3),dtype=int)
        Gss = torch.zeros((Ndt,Np,Np))  # Gaussian spread
        Pch,Cls = torch.zeros((Ndt,16,16,768)),torch.zeros((Ndt,768))
        for i in range(Ndt):
            # Get current image
            Tb2w = th.x_to_T(XbW[i,:])
            Tc2w = Tb2w@Tc2b
            iro,_ = gsplat.render_rgb(camera,Tc2w)

            # Process the image
            icr = process_image(iro)
            pch,cls = vit(icr.unsqueeze(0))  # Get the patch features

            # Get current overlay
            u,v = UVc[i,0], UVc[i,1]
            gss = np.zeros((Np,Np))
            for j in range(Np):
                for k in range(Np):
                    gss[j,k] = np.exp(-((pch_u[k]-u)**2 + (pch_v[j]-v)**2)/(2*sigma**2))
            gss = torch.from_numpy(gss).float()
            hmp = fu.heatmap_overlay(gss,iro)

            # Save the image data
            Img[i,:,:,:] = icr
            Hmp[i,:,:,:] = hmp
            Gss[i,:,:] = gss
            Pch[i,:,:,:] = pch
            Cls[i,:] = cls

        # Save the data 
        data = {
            "Xbw": XbW,"PtC": PtC,"UVc": UVc,
            "Img": Img,"Hmp": Hmp,"Gss": Gss,
            "Pch": Pch,"Cls": Cls,
        }
        data_path = os.path.join(folder, f"data{str(idx+1).zfill(3)}.pt")
        torch.save(data, data_path)

    # Sample some examples from the last dataset for visualization
    Nsp = 5
    idxs = np.random.choice(Ndt, Nsp, replace=False)

    _, axs = plt.subplots(nrows=Nsp, ncols=1, figsize=(5, Nsp * 1.5), sharex=True)
    for i, idx in enumerate(idxs):
        axs[i].imshow(data["Hmp"][idx])
    plt.tight_layout()
    plt.show()


def generate_poses(Ndt:int,p_tW:np.ndarray,
                   K:np.ndarray,Tb2c:np.ndarray,Rgl2cv:np.ndarray,
                   height:int,width:int) -> tuple[np.ndarray,np.ndarray]:
    
    # Some useful constants
    z_gW = np.array([0.0, 0.0, 1.0])
    k_rt = 0.20
    p_tW_h = np.hstack((p_tW, np.ones((1,))))

    # Generate the dataset
    XbW = np.zeros((Ndt, 10))
    PtC = np.zeros((Ndt, 3))
    UVc = np.zeros((Ndt, 2))
    for i in range(Ndt):
        # Sample a random point
        x_bW = np.random.uniform(-1.0, 1.0)
        y_bW = np.random.uniform(-1.0, 1.0)
        z_bW = np.random.uniform(-0.2,-2.0)

        p_bW = np.array([x_bW, y_bW, z_bW])

        # Compute rotation for target centered at p_tW
        xb = p_tW-p_bW
        xb = xb/np.linalg.norm(xb)
        yb = np.cross(z_gW,xb)
        zb = np.cross(xb, yb)
        Rw2b = np.vstack((xb, yb, zb))

        # Add noise to the rotation
        while True:
            delta = k_rt*np.random.randn(3)  # small angle
            Ri:np.ndarray = R.from_rotvec(delta).as_matrix()
            Rw2i = Ri.T@Rw2b

            # Compute uv
            Tb2w = np.eye(4)
            Tb2w[0:3,0:3] = Rw2i.T
            Tb2w[0:3,3] = p_bW
            Tw2b = np.linalg.inv(Tb2w)

            Tw2c = Tb2c@Tw2b
            p_tC_h = Tw2c@p_tW_h
            p_tC_h[0:3] = Rgl2cv@p_tC_h[0:3]  # Flip y and z for camera frame
            p_tC = p_tC_h[0:3]

            uv = K@p_tC[0:3]
            u,v = uv[0]/uv[2], uv[1]/uv[2]
            if 0 <= u < width and 0 <= v < height:
                # Pack the data
                XbW[i,0:3] = p_bW
                XbW[i,6:10] = R.from_matrix(Rw2i.T).as_quat()
                PtC[i,:] = p_tC
                UVc[i,0] = u
                UVc[i,1] = v

                break

    return XbW,PtC,UVc