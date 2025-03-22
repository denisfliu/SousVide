import numpy as np
from typing import List


def compute_flight_metrics(Trajectories:List[dict],pp_threshold:float=0.3) -> None:

    # Trajectory Tracking Error (TTE)
    TTE,TTE_best = [],np.inf
    for trajectory in Trajectories:
        # Extract trajectory data
        Pro:np.ndarray = trajectory["Xro"][0:3,:]
        Pds:np.ndarray = trajectory["tXd"][1:4,:]
        Ndata = trajectory["Ndata"]

        # Initialize the TTE for this trajectory
        tte = np.zeros(Ndata)

        # Compute the tte for each data point
        for i in range(Ndata):
            tte[i] = np.min(np.linalg.norm(Pro[:,i].reshape(-1, 1) - Pds, axis=0)) 
        TTE.append(tte)

        # Update the best tte if this trajectory has a better one
        tte_mean = np.mean(tte)
        if tte_mean < TTE_best:
            TTE_best = tte_mean
    
    TTE_all = np.hstack(TTE)
    TTE_mean = TTE_all.mean()

    # Proximity Percentile (PP)
    n_prox = np.sum(TTE_all < pp_threshold)
    pp = n_prox / len(TTE_all)
    
    # Inference Time
    T_inf = []
    for idx,trajectory in enumerate(Trajectories):
        t_inf = np.sum(trajectory["Tsol"], axis=0)
        T_inf.append(t_inf)

    hz_mean = 1/np.hstack(T_inf).mean()
    hz_worse = 1/np.hstack(T_inf).max()
    
    metrics = {
        "TTE": {
            "mean": TTE_mean,
            "best": TTE_best
        },
        "PP": pp,
        "hz": {
            "mean": hz_mean,
            "worse": hz_worse
        }
    }

    return metrics