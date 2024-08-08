#!/usr/bin/env python
# coding: utf-8

# In[7]:
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import h5py
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm


# In[8]:


NUM_NODES = 9


# In[15]:


def cossim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


# ## MSVD

# In[ ]:


NODE_FEATURES = './extracted_features/msvd/GridNodeFeatures.hdf5' # '<Path to the node features>.hdf5' # can be object-based or grid-based
SAVE_FILE = 'TemporalSimilarityGraph.hdf5' # "<Desired file name>.hdf5"


# In[16]:


with h5py.File(NODE_FEATURES, 'r') as fp, h5py.File(SAVE_FILE, 'w') as f:
    # loop through all MSVD video
    for vidid in tqdm(range(1, 1971)):
        video_id = 'vid'+str(vidid)
        for frid in range(20):
            curr_node_feat = fp['vid'+str(vidid)+'-'+str(frid)][:]
            if frid == 0:
                prev_node_feat = curr_node_feat
                continue
            # create temp zero tensor of num_nodes
            Gt_temp = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
            
            for k in range(len(prev_node_feat)):
                for l in range(len(curr_node_feat)):
                    if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                        continue
                        
                    # calculate the similarity between previous node and current node
                    Gt_temp[k][l] = math.exp(cossim(prev_node_feat[k], curr_node_feat[l]))
                    if np.isnan(Gt_temp[k][l]):
                        print(prev_node_feat[k])
                        print(curr_node_feat[l])
                        
            Gt = [[0.0] * NUM_NODES for i in range(NUM_NODES)]
            for k in range(len(prev_node_feat)):
                for l in range(len(curr_node_feat)):
                    if (np.sum(prev_node_feat[k])==0 or np.sum(curr_node_feat[l])==0):
                        continue
                    Gt[k][l] = Gt_temp[k][l]/sum(Gt_temp[k])
                
            
            prev_node_feat = curr_node_feat
            f.create_dataset(video_id+'-'+str(frid), data = Gt)         
