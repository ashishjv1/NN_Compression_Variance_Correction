import numpy as np
# from flopco import FlopCo
import sys
from collections import defaultdict
import copy
import gc

import torch
# from stable_lr_cnn_compression.utils.replacement_utils import replace_conv_layer_by_name
# from decompose import get_decomposed_layer


def estimate_rank_for_compression_rate(tensor_shape, tensor,
                                       rate=2.,
                                       key='tucker2'):
    '''
        Find max rank for which inequality (initial_count / decomposition_count > rate) holds true
    '''
    min_rank = 3
    max_rank = min_rank

    initial_count = np.prod(tensor_shape)
    
#     if key != 'svd':
#         cout, cin, kh, kw = tensor_shape
        
    if key == 'tucker2':
        # tucker2_rank when R4=beta*R3
        cout, cin, kh, kw = tensor_shape
        if cout != cin:
            beta = cout/cin
        else:
            beta = 1.

        a = 1
        b = (cin + beta * cout) / (beta * kh * kw)
        c = -cin * cout / rate / beta

        discr = b ** 2 - 4 * a * c
        max_rank = int((-b + np.sqrt(discr)) / 2 / a)
        # [R4, R3]

        max_rank = max(max_rank, min_rank)
        max_rank = (max_rank, int(beta * max_rank))

#     elif key == 'svd':
#         max_rank = initial_count // (rate * sum(tensor_shape[:2]))
#         max_rank = max(max_rank, min_rank)
        
    elif key == 'matrix':
        max_rank = np.linalg.matrix_rank(tensor)
    
    return max_rank