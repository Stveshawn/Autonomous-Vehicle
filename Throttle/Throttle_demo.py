#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from data_pre import *
from Throttle import *


# In[2]:


data = load_data()


# In[3]:


m = Throttle(data)
m.update_vars(m.data)
m.filter_obs()


# In[4]:


m.update_vars(m.data2)
m.fit_by_batches()


# In[5]:


# Estimated parameters for the first 5 batches
m.mus[:5], m.Sigs[:5]


# In[6]:


m.conditional_dist()


# In[8]:


# take the most recent data
data_base = m.data2.copy()[int(36*m.batch_size):][['v', 'cmd']]
m.outlier_detect(data_base)

