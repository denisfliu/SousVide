import os
import torch
import numpy as np
import figs.utilities.config_helper as ch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from figs.simulator import Simulator
from sousvide.control.networks.pave import Pave
from sousvide.control.pilot import Pilot


class PPO:
    def __init__(self,
                 cohort_name:str,gsplat_name:str,method_name:str,
                 pilot:str,Neps:int=100,Ndata:int=128,Nppo:int=300,
                 frame_name:str="carl"):

        # Some useful constants
        np.random.seed(42)  # For reproducibility
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x1 = torch.Tensor([
            0.00, 0.00, 0.00,           # test1 position
            0.00, 0.00, 0.00,           # test1 velocity
            0.00, 0.00, 0.00, 1.00      # test1 orientation (quaternion)
        ]).unsqueeze(0).to(device)
        x2 = torch.Tensor([
            0.00, 0.00,-2.00,           # test2 position
            0.00, 0.00, 0.00,           # test2 velocity
            0.00, 0.00, 0.00, 1.00      # test2 orientation (quaternion)
        ]).unsqueeze(0).to(device)
        x3 = torch.Tensor([
            0.00, 0.00,-1.00,           # test2 position
            0.00, 0.00, 0.00,           # test2 velocity
            0.00, 0.00, 0.00, 1.00      # test2 orientation (quaternion)
        ]).unsqueeze(0).to(device)
        Xnn1 = {"current": x1}
        Xnn2 = {"current": x2}
        Xnn3 = {"current": x3}
        
        # Extract configs
        gsplat = ch.get_gsplat(gsplat_name)
        bframe = ch.get_config(frame_name,"frames")
        method = ch.get_config(method_name,"methods")

        # Initialize the simulator
        self.simulator = Simulator(gsplat,method,bframe)
        
        # Initialize the pilot
        self.controller = Pilot(cohort_name,pilot)
        self.controller.update_frame(bframe)

        policy:Pave = self.controller.policy.networks["commNet"]

        # Test initial output
        Ynn1 = policy(Xnn1)
        Ynn2 = policy(Xnn2)
        Ynn3 = policy(Xnn3)
        print("Initial Action 1:", Ynn1["command"].detach())
        print("Initial Action 2:", Ynn2["command"].detach())
        print("Initial Action 3:", Ynn3["command"].detach())

        # Run PPO training
        Losses = []
        for i in range(Neps):
            # Generate Rollout data
            Obs_buff, Act_buff, LPb_buff, Rtg_buff, Adv_buff = [], [], [], [], []
            for _ in range(Ndata):
                # Generate a rollout
                Obs,Act,LPb,Rtg,Adv = self.generate_rollout()

                Obs_buff.extend(Obs)
                Act_buff.extend(Act)
                LPb_buff.extend(LPb)
                Rtg_buff.extend(Rtg)
                Adv_buff.extend(Adv)

            # PPO update
            Losses_i = self.ppo_update(
                Obs_buff,Act_buff,LPb_buff,Rtg_buff,Adv_buff,
                clip_eps=0.2, epochs=Nppo, batch_size=64
            )

            # Update losses
            Losses.extend(Losses_i)

            # Save the policy
            student_path = self.controller.path

            netw_path = os.path.join(student_path,"commNet.pt")
            torch.save(policy,netw_path)

            # Print progress
            print(f"Completed Epoch {i+1}, Loss: {Losses_i[-1]:.4f}")

        print("Training complete.")
                    
        # Plot the losses
        plt.plot(Losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Test learned output
        Ynn1 = policy(Xnn1)
        Ynn2 = policy(Xnn2)
        Ynn3 = policy(Xnn3)
        print("Final Action 1:", Ynn1["command"].detach())
        print("Final Action 2:", Ynn2["command"].detach())
        print("Final Action 3:", Ynn3["command"].detach())
        
    def generate_rollout(self):

        # Some useful constants
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xi = np.array([
            0.00, 0.00,-1.00,           # Initial position
            0.00, 0.00, 0.00,           # Initial velocity
            0.00, 0.00, 0.00, 1.00      # Initial orientation (quaternion)
        ])
        w0 = np.array([
            0.20, 0.20, 0.20,           # Position noise
            0.20, 0.20, 0.20,           # Velocity noise
            0.00, 0.00, 0.00, 0.00      # Orientation noise
        ])

        # Some useful constants
        pt = torch.Tensor(xi[0:3]).to(device)   # Target position
        vt = torch.Tensor(xi[3:6]).to(device)   # Target position and velocity
        tf_dat,tf_hzn = 5.0,1.0                 # Data and horizon timeframes
        tf_tot = tf_dat + tf_hzn                # Total time for rollout
        Ndat = int(tf_dat*20)                   # Number of data points in the trajectory
        Ntot = int(tf_tot*20)                   # Total number of points in the trajectory
        gamma = 0.99                            # Discount factor for return-to-go
        
        # Generate a single dynamic rollout
        x0 = xi + np.random.uniform(-w0,w0)
        x0[6:10] = x0[6:10] / np.linalg.norm(x0[6:10])
        
        self.controller.reset_memory(x0)
        _,Xro,Uro,_,_,_,_ = self.simulator.simulate(self.controller,0.0,tf_tot,x0)

        # Convert to torch tensors
        Xro = torch.from_numpy(Xro).float().to(device)
        Uro = torch.from_numpy(Uro).float().to(device)

        # Initialize buffers
        Obs_buff = []   # Buffer for observations
        Act_buff = []   # Buffer for actions
        LPb_buff = []   # Buffer for log probabilities
        Val_buff = []   # Buffer for values
        Adv_buff = []   # Buffer for advantages

        # Compute RL rollout
        actor:Pave = self.controller.policy.networks["commNet"]
        for i in range(Ndat):
            Xnn = {"current":Xro[i,:].unsqueeze(0)}
            with torch.no_grad():
                Ynn = actor.get_action(Xnn)

            Obs_buff.append(Xnn)
            Act_buff.append(Ynn["action"])
            LPb_buff.append(Ynn["l_prob"])
            Val_buff.append(Ynn["ct_val"])

        # Calculate the return-to-go (RTG)
        Rew_buff = []   # Buffer for rewards
        for i in range(Ntot):
            pi, vi = Xro[i, 0:3], Xro[i, 3:6]  # Position and velocity
            reward = -torch.linalg.norm(pi - pt) - 0.1 * torch.linalg.norm(vi - vt)  # Reward function
            Rew_buff.append(reward)

        Rtg_buff,rtg = [],0
        for r in reversed(Rew_buff):
            rtg = r + gamma*rtg
            Rtg_buff.insert(0,rtg)

        Rtg_buff = Rtg_buff[:Ndat]

        # Caclulate the advantage
        Adv_buff = []
        for i in range(Ndat):
            adv = Rtg_buff[i] - Val_buff[i]
            Adv_buff.append(adv)

        # Normalize the advantages
        Adv_buff = torch.tensor(Adv_buff, dtype=torch.float32)
        Adv_buff = (Adv_buff - Adv_buff.mean()) / (Adv_buff.std() + 1e-8)
        Adv_buff = Adv_buff.tolist()

        return Obs_buff, Act_buff, LPb_buff, Rtg_buff, Adv_buff
    
    def ppo_update(self,
                   Obs:torch.Tensor,Act:torch.Tensor,
                   LPb:torch.Tensor,Rtg:torch.Tensor,Adv:torch.Tensor,
                   clip_eps=0.2, epochs=100, batch_size=64):
        
        # Some useful constants
        Ndt = len(Obs)
        idxs = np.arange(Ndt)

        # PPO update variables
        policy:Pave = self.controller.policy.networks["commNet"]
        optimizer = optim.Adam(policy.parameters())
    
        Losses = []
        for epoch in range(epochs):
            np.random.shuffle(idxs)

            Loss = 0
            for start in range(0, Ndt, batch_size):
                end = start + batch_size
                batch_idxs = idxs[start:end]

                obs_buf = torch.stack([Obs[i]["current"].squeeze(0) for i in batch_idxs]).to(torch.float32)
                logp_buf = torch.stack([LPb[i] for i in batch_idxs]).to(torch.float32).squeeze(-1)
                ret_buf = torch.tensor([Rtg[i] for i in batch_idxs], dtype=torch.float32).to(obs_buf.device)
                adv_buf = torch.tensor([Adv[i] for i in batch_idxs], dtype=torch.float32).to(obs_buf.device)

                # New action & log-prob under current policy
                Xnn = {"current": obs_buf}
                Ynn = policy.get_action(Xnn)
                new_l_prob = Ynn["l_prob"]
                
                # Calculate the ratio of new to old probabilities
                ratio = torch.exp(new_l_prob - logp_buf)

                # Clipped surrogate loss
                surr1 = ratio * adv_buf
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_buf
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                value_pred = Ynn["ct_val"].squeeze(-1)
                value_loss = F.mse_loss(value_pred, ret_buf)

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss

                # Accumulate the loss
                Loss += total_loss.item()/ batch_size

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            Losses.append(Loss)

        return Losses

        #         # Update the policy
        #         self.update_policy(obs_buf, act_buf, logp_buf, ret_buf, adv_buf, clip_eps, batch_size)
        
        # for _ in range(epochs):
        #     indices = np.random.permutation(dataset_size)
        #     for start in range(0, dataset_size, batch_size):
        #         end = start + batch_size
        #         batch_idx = indices[start:end]

        #         obs_b = obs_buf[batch_idx].to(device)
        #         act_b = act_buf[batch_idx].to(device)
        #         logp_old_b = logp_buf[batch_idx].to(device)
        #         ret_b = ret_buf[batch_idx].to(device)
        #         adv_b = adv_buf[batch_idx].to(device)

        #         # New action & log-prob under current policy
        #         Xnn = {"current":obs_b}
        #         Ynn = actor(Xnn)
        #         new_act = Ynn["command"]
        #         new_logp = Ynn["log_prob"]

        #         ratio = torch.exp(new_logp - logp_old_b)

        #         # PPO loss
        #         surr1 = ratio * adv_b
        #         surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
        #         policy_loss = -torch.min(surr1, surr2).mean()

        #         # Critic loss
        #         value_pred = Ynn["value"]
        #         value_loss = F.mse_loss(value_pred, ret_b)

        #         loss = policy_loss + 0.5 * value_loss

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()