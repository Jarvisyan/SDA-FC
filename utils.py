import torch
import random
import numpy as np
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()      

def gen_cond_label(batch_size, class_num, z_dim):
    conditional_label = torch.zeros(batch_size, class_num)
    cluster_size = round(batch_size / class_num)
    for i in range(class_num):
        if i == class_num - 1:
            conditional_label[i * cluster_size : , i] = 1
        else:
            conditional_label[i * cluster_size : (i + 1) * cluster_size, i] = 1
    G_input = torch.cat([conditional_label, torch.randn(batch_size, z_dim)], 1)
    return G_input, conditional_label
    
def construct_clients(Nets, class_num, p, images, ground_truth):
    if p == 1:
        for i in range(class_num):
            Nets[f'X_{i}'] = images[ground_truth == i]
            Nets[f'y_{i}'] = ground_truth[ground_truth == i]
    else:
        n = int(images.shape[0] / class_num) #The number of samples allocated to each client.
        idx_rest = np.zeros(0, int)
        
    # Each client selects n * p samples from each class.
    for i in range(class_num): 
        Nets[f'd_{i}'] = np.where(ground_truth == i)[0][ : round(n * p)] #Indices of the selected samples in class i.
        d_i_rest = np.where(ground_truth == i)[0][round(n * p) : ] #Indices of the unselected samples in class i.
        idx_rest = np.concatenate((idx_rest, d_i_rest))
    
    # Each client selects n * (1 - p) samples from the unselected samples randomly
    shuffle_idx = torch.randperm(idx_rest.shape[0])
    idx_rest_shuffled = idx_rest[shuffle_idx] #Shuffle the indices of the unselected samples.
    idx1, idx2 = 0, round(n * (1-p))
    for i in range(class_num):
        Nets[f'd_{i}'] = np.concatenate((Nets[f'd_{i}'], idx_rest_shuffled[idx1 : idx2]))
        idx1 = idx2
        idx2 += round(n * (1-p))
      
    for i in range(class_num):
        Nets[f'X_{i}'] = images[Nets[f'd_{i}']]
        Nets[f'y_{i}'] = ground_truth[Nets[f'd_{i}']]
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True