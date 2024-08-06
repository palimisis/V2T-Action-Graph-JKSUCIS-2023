#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
import glob
import json
import h5py
import math
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as trn
import torchvision.models as models
import torchvision.ops.roi_align as roi_align

from c4c_modules.until_module import PreTrainedModel, AllGather, CrossEn
from c4c_modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from c4c_modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pickle
import pathlib

from dataloaders.dataloader_msrvtt import MSRVTT_RawDataLoader
from dataloaders.dataloader_msvd import MSVD_Loader

# In[2]:


from c4c_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from c4c_modules.modeling import CLIP4Clip
from c4c_modules.optimization import BertAdam
from util import parallel_apply, get_logger

device = torch.device('cuda')


# In[19]:


# Argument
class args:
    msvd = True # or msvd = False for MSR-VTT
    slice_framepos=2
    dset ='./dataset' # change based on dataset location
    save_path = './extracted_features'
    max_frames = 20
    eval_frame_order = 0 
    output_dir='./feature_extractor/pretrained'
    cache_dir=''
    
    pretrained_clip_name = "ViT-B/16"

    features_path='..'
    msrvtt_csv ='msrvtt.csv'
    data_path ='MSRVTT_data.json'
    max_words=32
    feature_framerate=1
    cross_model="cross-base"
    local_rank=0


# In[20]:


#MSVD 
if args.msvd:
    dset_path = os.path.join(os.path.join(args.dset),'MSVD')
    features_path = os.path.join(dset_path,'raw') # video .avi    
    name_list = glob.glob(features_path+os.sep+'*')
    args.features_path = features_path

    url2id = {}
    data_path = os.path.join(os.path.join(dset_path,'captions','youtube_mapping.txt'))
    args.data_path = data_path
    for line in open(data_path,'r').readlines():
        url2id[line.strip().split(' ')[0]] = line.strip().split(' ')[-1]

    path_to_saved_models = f"{args.save_path}/msvd"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/MSVD_Clip4Clip_features.pickle'
    args.max_words =30
    
    
    videos= MSVD_Loader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 0,
    ) 
#MSR-VTT    
else:
  
    dset_path = os.path.join(os.path.join(args.dset,'dataset'),'MSRVTT')
    features_path = os.path.join(dset_path,'raw') 
    args.features_path = features_path
    data_path=os.path.join(dset_path,'MSRVTT_data.json')
    args.data_path = data_path
    args.msrvtt_csv = os.path.join(dset_path,'msrvtt.csv')
    name_list = glob.glob(features_path+os.sep+'*')
    
    path_to_saved_models = "extracted/msrvtt"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    save_file = path_to_saved_models+'/MSRVTT_Clip4Clip_features.pickle'
    args.max_words =73
    
    #Load video to dataloader
    videos= MSRVTT_RawDataLoader(
        csv_path=args.msrvtt_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        transform_type = 0,
)


# ### Load the CLIP4Clip pretrained models

# In[21]:


epoch = 5
model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch-1))
model_state_dict = torch.load(model_file, map_location='cuda')
cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)



# In[23]:


clip = model.clip.to(device)


# ### Extract clip features

# In[24]:

clip.eval()

with torch.no_grad():
    data ={}
    stop = False
    with open(save_file, 'wb') as handle:

        for i in tqdm(range(len(videos))):

            video_id,video,video_mask = videos[i]

            tensor = video[0]


            tensor = tensor[video_mask[0]==1,:]

            tensor = torch.as_tensor(tensor).float()
            video_frame,num,channel,h,w = tensor.shape
            tensor = tensor.view(video_frame*num, channel, h, w)

            video_frame,channel,h,w = tensor.shape


            output = clip.encode_image(tensor.to(device), video_frame=video_frame).float().to(device)

            output = output.detach().cpu().numpy()
            data[video_id]=output

            del output
         
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:




