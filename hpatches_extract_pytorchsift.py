import sys
import argparse
import torch
import time
import os
import sys
import cv2
import math
import numpy as np
from tqdm import tqdm
from pytorch_sift import SIFTNet
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os

assert len(sys.argv)==3, "Usage python hpatches_extract_sift.py hpatches_db_root_folder 64"
OUT_W = int(sys.argv[2])    
# all types of patches 
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))
            
    
seqs = glob.glob(sys.argv[1]+'/*')
seqs = [os.path.abspath(p) for p in seqs]     

descr_name = 'pytorch-sift-'+str(OUT_W)


model = SIFTNet(OUT_W, masktype = 'Gauss')
model.cuda()

for seq_path in seqs:
    seq = hpatches_sequence(seq_path)
    path = os.path.join(descr_name,seq.name)
    if not os.path.exists(path):
        os.makedirs(path)
    descr = np.zeros((seq.N,128))
    for tp in tps:
        print(seq.name+'/'+tp)
        if os.path.isfile(os.path.join(path,tp+'.csv')):
            continue
        n_patches = 0
        for i,patch in enumerate(getattr(seq, tp)):
            n_patches+=1
        t = time.time()
        patches_for_net = np.zeros((n_patches, 1, OUT_W, OUT_W))
        uuu = 0
        if OUT_W != 65:
            for i,patch in enumerate(getattr(seq, tp)):
                patches_for_net[i,0,:,:] = cv2.resize(patch,(OUT_W,OUT_W))
        else:
            for i,patch in enumerate(getattr(seq, tp)):
                patches_for_net[i,0,:,:] = patch      
        ###
        model.eval()
        outs = []
        bs = 128;
        n_batches = n_patches / bs + 1
        for batch_idx in range(n_batches):
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > n_patches:
                    end = n_patches
                else:
                    end = (batch_idx + 1) * bs
            else:
                end = (batch_idx + 1) * bs
            data_a = patches_for_net[batch_idx * bs: end, :, :, :].astype(np.float32)
            data_a = torch.from_numpy(data_a)
            data_a = data_a.cuda()
            with torch.no_grad():
                out_a = model(data_a)
            outs.append(out_a.data.cpu().numpy().reshape(-1, 128))
        res_desc = np.concatenate(outs)
        res_desc = np.reshape(res_desc, (n_patches, -1))
        out = np.reshape(res_desc, (n_patches,-1))
        outs = np.clip(512*out, 0, 255)
        np.savetxt(os.path.join(path,tp+'.csv'), outs.astype(np.uint8), delimiter=',', fmt='%d')
