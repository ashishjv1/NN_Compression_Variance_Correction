import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import copy
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms
from decomposer import decompose
from torch.autograd import Variable
import tensorly as tl
from tensorly.random import check_random_state
from select_ranks import estimate_rank_for_compression_rate
from replacement_utils import get_layer_by_name, replace_conv_layer_by_name
from tkd_layers import TKD2_layer
from tqdm import tqdm



def replace_layers(lnames, model, compressed_model, ranks, factors):
    
    print("Compressing layer:", lnames)
    layer_to_decompose = get_layer_by_name(model, lnames)
    decomposed_layer = TKD2_layer(layer_to_decompose, ranks, factors)   
    replace_conv_layer_by_name(compressed_model, lnames, decomposed_layer)

    return compressed_model




def initialize(weights, cr=0.2, modes=[0,1], key="tucker2"):
    
    random_state = 1234
    rng = check_random_state(random_state)
    
    if key == "tucker2":
        rank = estimate_rank_for_compression_rate(weights.shape, weights, rate=1/cr, key=key) + (weights.shape[2], weights.shape[3])
        
        core = Variable(torch.nn.init.xavier_normal_(torch.tensor(rng.random_sample(rank), **tl.context(weights))),
                        requires_grad=True)
        
        factors = [Variable(torch.nn.init.xavier_normal_(
            torch.tensor(rng.random_sample((tl.shape(weights)[mode], rank[index])),
                         **tl.context(weights))), requires_grad=True) for (index, mode) in enumerate(modes)]
        return rank, core, factors
    
    else:
        
        rank = estimate_rank_for_compression_rate(weights.squeeze().shape, weights.squeeze(), rate=1/cr, key=key)
        
        factors = [Variable(torch.nn.init.xavier_normal_(tl.tensor(rng.random_sample((weights.shape[i], rank)))), 
                            requires_grad=True) for i in range(weights.ndim)]
        
        core = None
    
        return rank, core, factors



def compress_layers(model, key='tucker2', cr=0.2, modes=[0,1], n_iter=5000, lr=0.0002, penalty=0.7, print_iter=1000):
    
    compressed_model = copy.deepcopy(model).cpu()
    lnames_to_compress = [module_name for module_name, module in model.named_modules()
                          if isinstance(module, nn.Conv2d)]
    
    for lname in tqdm(lnames_to_compress):
        layer_to_decompose = get_layer_by_name(model, lname)
        weights = layer_to_decompose.weight.data
#         print(key)
        rank, core, factors = initialize(weights,cr=0.2, modes=[0,1], key=key)
        
        TD_NET = decompose(weights, factors=factors, core=core, n_iter=n_iter, lr=lr, penalty=penalty, print_iter = print_iter)
        TD_NET.train()
        
        factor = TD_NET.core, TD_NET.factors
        
        replace_layers(lname, model, compressed_model, rank, factor)
        
    return TD_NET, compressed_model


