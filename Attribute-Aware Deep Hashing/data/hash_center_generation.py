from scipy.special import comb, perm  #calculate combination
from itertools import combinations
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import mindspore as ms
import numpy as np

dataset_type = "nus_wide"
d = 64 # d is the lenth of hash codes and hash centers, d should be 2^n
ha_d = hadamard(d)   # hadamard matrix 
ha_2d = np.concatenate((ha_d, -ha_d),0)  # can be used as targets for 2*d hash bit

num_class = 21
if num_class<=d:
    hash_targets = torch.from_numpy(ha_d[0:num_class]).float()
    print('hash centers shape: {}'. format(hash_targets.shape))
elif num_class>d:
    hash_targets = torch.from_numpy(ha_2d[0:num_class]).float()
    print('hash centers shape: {}'. format(hash_targets.shape))

# Save the hash targets as training targets
file_name = str(d) + '_' + dataset_type + '.pkl'
file_dir = f'data/{dataset_type}/hash_centers/' + file_name
f = open(file_dir, "wb")
torch.save(hash_targets, f)