#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Some useful settings for interactive work
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# get_ipython().run_line_magic('matplotlib', 'widget')

import torch
torch.set_float32_matmul_precision('high')


# In[2]:


# Import the relevant modules
import sousvide.synthesize.rollout_generator as rg
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_learning as pl
import sousvide.flight.deploy_figs as df


# In[3]:


cohort = "test2"

data_method = "test2_data"
data_method = "test2_data_f2"
eval_method = "test2_eval"

scene = "mid_gate"

courses = [
    "traverse"
    "traverse_f2",
    ]   

roster = [
    "hsCameron",
    "hsDavion",
    ]


# In[ ]:


# # Generate Rollouts
# rg.generate_rollout_data(cohort,["traverse_f2"],scene,"test2_data_f2")
# rg.generate_rollout_data(cohort,["traverse"],scene,"test2_data")

# # Review the Rollout Data
# ps.plot_rollout_data(cohort)


# In[ ]:


# # Generate initial observation data sets
# og.generate_observation_data(cohort,roster)


# In[ ]:


# # Train the Policy
# tp.train_roster(cohort,roster,"histNet",100)

# pl.plot_losses(cohort,roster,"histNet",use_log=True)


# In[ ]:


# Train the Policy
# tp.train_roster(cohort,roster,"commNet",150)
tp.train_roster(cohort,roster,"commNet",200,regen=True,
                use_deploy=scene,deploy_method=eval_method,lim_sv=50)

pl.plot_losses(cohort,roster,"commNet",use_log=True)


# In[ ]:


# Simulate in FiGS
for course in courses:
    df.deploy_roster(cohort,course,scene,eval_method,roster,mode="visualize")
    pl.plot_deployments(cohort,course,roster,plot_show=True)

