#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import torch
import pickle
import itertools



from tqdm import tqdm
from torch_geometric.data import Data
from utility.util import stack_node_features

import pathlib
import numpy as np
import sparse
import gc


# ## Data Preparation

# In[ ]:


# Arguments
class args:
    dataset = 'msvd' # or dataset = 'msvd'


# In[ ]:


# Set configuration
path_to_saved_models = "extracted_features/"+args.dataset
pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
stg_file =path_to_saved_models+'/ActionSpatioTemporalGraph.hdf5' # '/<Path to spatio temporal graph>.hdf5'
fo_file =path_to_saved_models+'/GridNodeFeatures.hdf5' # '/<Path to node features>.hdf5'


# In[ ]:


# Prepare action data
stg = []
with (open(stg_file, "rb")) as openfile:
    while True:
        try:
            stg.append(pickle.load(openfile))
            if len(stg)==10000:
                break
        except EOFError:
            break


# # Prepare graph feature

# In[ ]:


# Stack object feature
fo = stack_node_features(fo_file)


# In[ ]:


def generate_graph_data(stg_vid, fo_vid):
    """Generate graph data for every vid_id STG & FO"""
    t =[]
    attr =[]
    n_rows = stg_vid.shape[0]
    n_columns = stg_vid.shape[1]
    n_dim_feature = stg_vid.shape[2]
    n_dim_fo = fo_vid.shape[1]
    
    allzero = False
    
    # Edge index
    edge_index = torch.tensor(list(map(list, itertools.product(np.arange(n_rows), repeat=2))), dtype=torch.long)
      
    # Edge feature
    edge_attr = torch.tensor(stg_vid.todense()[:n_rows, :n_columns], dtype=torch.float).reshape(n_rows * n_columns, n_dim_feature)

    for i in range (len(edge_attr)):
        allzero = torch.sum(edge_attr[i])
        if allzero > 0:
            t.append(edge_index[i])
            attr.append(edge_attr[i])

    # Node feature
    if(len(t)==0):
        v=edge_index[0].unsqueeze(0)
        attr = edge_attr[0].unsqueeze(0)
        allzero = True
    else:
        v = torch.stack(t)
        attr = torch.stack(attr)
   
    x = torch.tensor(fo_vid[:n_rows], dtype=torch.float)


    # Generate the graph
    data = Data(x=x, edge_index=v.t().contiguous(), edge_attr=attr)
    del attr
    del v
    del t
    
    return data,allzero


# In[ ]:


# Sort the STGraph vid_id
if args.dataset == 'msvd':
    ids = [int(list(x.keys())[0].split("vid")[1]) for x in stg]
else:
    ids = [int(list(x.keys())[0].split("video")[1]) for x in stg] 
contents = stg.copy()
stg = [x for _,x in sorted(zip(ids,contents))]

stg = stg[:985]


# In[ ]:


# Generate Pytorch geometric data
datas = {}
index=[]
for i in tqdm(range(len(stg))):
    if args.dataset == 'msvd':
        id = 'vid' + str(i+1)
    else:
        id = 'video' + str(i)
    stg_vid = stg[i][id]

    fo_vid = fo[id]

    datas[id],allzero = generate_graph_data(stg_vid, fo_vid)
    if allzero:
        index.append(i)


# ## Generate Pytorch Geometric-based Graph Structure

# In[ ]:


# Save memory by deleting previous data (if the cell has been run multiple times)
stg = None
fo = None
contents = None
ids = None
del stg
del fo
del contents
del ids
gc.collect()


# In[ ]:


# Generate the data structure
num_object = 9
num_edge_features = 1024
num_frame = 20
num_node = num_object*num_frame
max_ = 0

for key in datas.keys():
    datas[key].edge_attr = sparse.COO(np.array(datas[key].edge_attr))

for g in datas:
    max_ = max(datas[g].edge_index.shape[1], max_)
    
hmap = {}
for g in tqdm(datas):
    for i in range(datas[g].edge_index.shape[1]):
        key = str(g)+'-'+str(datas[g].edge_index[0][i].item())+'-'+str(datas[g].edge_index[1][i].item())
        hmap[key] = 1

for g in tqdm(datas):
    curr_size = datas[g].edge_index.shape[1]
    
    if curr_size < max_:
        counter = max_ - curr_size
        done = False
        if type(datas[g].edge_attr)!=np.ndarray:
            datas[g].edge_attr = datas[g].edge_attr.todense()
        for i in range(num_node):
            for j in range(num_node):
                key = str(g)+str(i)+'-'+str(j)
                if (key in hmap) == False:
                    datas[g].edge_index = torch.hstack((datas[g].edge_index, torch.tensor([[i],[j]])))
                    datas[g].edge_attr = np.vstack((datas[g].edge_attr,np.zeros(num_edge_features)))
                    counter -= 1
                    
                    if counter==0:
                        done =True
                        break
            if done:
                break
        datas[g].edge_attr = sparse.COO(datas[g].edge_attr)
        


# In[ ]:


# Save
action_graph = path_to_saved_models+'/FinalGraph_1.pickle' # '/<desired file name>.pickle'
with open(action_graph, 'wb') as fp:
     pickle.dump(datas, fp)
        
print("SPATIO TEMPORAL ACTION GRAPH SUCCESSFULLY SAFE")

