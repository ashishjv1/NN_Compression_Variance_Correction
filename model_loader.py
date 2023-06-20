import os
import model_loader_cifar

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = model_loader_cifar.load(model_name, model_file, data_parallel)
    return net
