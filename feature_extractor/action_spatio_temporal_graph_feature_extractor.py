#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import h5py
from tqdm import tqdm
import torch
import pathlib
import sparse
import pickle

device = torch.device('cuda')


# ## Generate Spatio-Temporal Graph

# #### Path Specification

# In[2]:


# Argument
class args:
    dataset = 'msvd' # or dataset = 'msrvtt'    
    num_edge_feat = 1024 # Dimension of edge feature


# In[3]:


# Output ST Graph path
path_to_saved_models = "extracted_features/"+args.dataset
pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)


# In[6]:


# Load spatial and temporal features
try:
    sf_file =path_to_saved_models+'SpatialActionGraph.hdf5' # '/<Path to spatial graph>.hdf5'
    ft_file =path_to_saved_models+'TemporalSimilarityGraph.hdf5' # '/<Path to temporal graph>.hdf5'
    
    save_file = path_to_saved_models+'ActionSpatioTemporalGraph.hdf5' # '/<Desired filename>.hdf5'
    
    fs = h5py.File(sf_file,'r')
    ft = h5py.File(ft_file,'r')
except Exception as e:
    print(e)


# #### Preprocess Values From Spatial and Temporal Feature

# In[7]:


# Load values from spatial graph
num_object = 9

sgraph_list = {}
tgraph_list = {}
count = 0
for i,key in tqdm(enumerate(fs.keys()), total=len(fs.keys())):
    a = key.split('-')

    sgraph = fs[key][:]
    temp=[]
    for k in range(num_object):
        for l in range(num_object):
            if isinstance (sgraph[k][l], str):
                sgraph[k][l] = eval(sgraph[k][l])

            else:
                sgraph[k][l].astype(np.float64)
                    
    if key in ft:
        tgraph = ft[key][:]
        if a[0] not in tgraph_list:
            tgraph_list[a[0]] = {}
        tgraph_list[a[0]][int(a[1])] = tgraph
    if a[0] not in sgraph_list:
        sgraph_list[a[0]] = {}
    sgraph_list[a[0]][int(a[1])] = sgraph


# In[8]:


# Load values from temporal graph
for a in tgraph_list:
    tgraph = tgraph_list[a]
    for b in tgraph:
        tgraph[b] = np.concatenate((np.expand_dims(tgraph[b], axis=2),np.zeros((num_object,num_object,1023))), axis=2)


# #### STGraph File Generation

# In[ ]:


num_object = 9 # Number of patches
num_frame = 20 # Num of frames

mpzeros = np.zeros((num_frame*num_object, num_frame*num_object, args.num_edge_feat))
with open(save_file, 'ab+') as handle:
    for k in tqdm(sgraph_list.keys(), total=len(sgraph_list.keys())):
        mgraph = mpzeros
        sorted_k = sorted(sgraph_list[k].keys())
        for i,k_fr in enumerate(sorted_k):
            s_start = i*num_object
            s_end = (i*num_object)+num_object
            t_start = s_start+num_object
            t_end = s_start+(num_object*2)
            
            mgraph[s_start:s_end,s_start:s_end] = sgraph_list[k][k_fr]
            if i<len(sorted_k)-1:
                mgraph[s_start:s_end, t_start:t_end] =tgraph_list[k][sorted_k[i+1]]
        s = {}
        y = sparse.COO(mgraph)
        s = {k:y}
        pickle.dump(s, handle)

