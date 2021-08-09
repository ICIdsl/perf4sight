import os
import sys
import copy
import glob
import time
import pickle
import importlib
import subprocess

import torch

# get current directory and append to path
# this allows everything inside pruners to access anything else 
# within pruners regardless of where pruners is
currDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currDir)

import pruners.base as pruningSrc
from pruners.vgg import VGGPruning
from pruners.resnet import ResNetPruning
from pruners.alexnet import AlexNetPruning
from pruners.mnasnet import MnasNetPruning 
from pruners.googlenet import GoogLeNetPruning 
from pruners.squeezenet import SqueezeNetPruning 
from pruners.mobilenetv2 import MobileNetV2Pruning 

def get_model(params, network):
#{{{
    modelfile = params.pruner['model_desc_dir']
    
    path = modelfile.replace('/','.')[:-1]
    module = importlib.import_module(path)
    
    if 'resnet' in network:
        depth = network.split('resnet')[1] # passing in resnet50 will extract depth as 50
        assert depth != '', f"Specify resnet depth as resnet50/resnet18 etc."
        params.arch = 'resnet'
        params.depth = depth
        model = module.__dict__['resnet'](num_classes=1000, pretrained=False, depth=int(depth))
    else:
        params.arch = network
        model = module.__dict__[network](num_classes=1000, pretrained=False)
    
    model = torch.nn.DataParallel(model)
    return model
#}}}

def setup_pruners(params, model):
#{{{
    params.dataset = 'imagenet'
    
    if 'alexnet' in params.arch:
        pruner = AlexNetPruning(params, model)
    elif 'resnet' in params.arch:
        pruner = ResNetPruning(params, model)
    elif 'mobilenet' in params.arch:
        pruner = MobileNetV2Pruning(params, model)
    elif 'squeezenet' in params.arch:
        pruner = SqueezeNetPruning(params, model)
    elif 'vgg' in params.arch:
        pruner = VGGPruning(params, model)
    elif 'mnasnet' in params.arch:
        pruner = MnasNetPruning(params, model)
    elif 'googlenet' in params.arch:
        pruner = GoogLeNetPruning(params, model)
    else:
        raise ValueError("Pruning not implemented for architecture ({})".format(params.arch))

    return pruner
#}}}

def create_dataset(params):
#{{{
    networks = eval(params.pruner['networks'])
    pruningLevels = copy.deepcopy(eval(params.pruner['pruning_perc']))
    for net in networks:
        model = get_model(params, net)
        assert model is not None

        origModel = copy.deepcopy(model)
        for pp in pruningLevels:
            print(f"Pruning network {net} at pruning level {pp}")
            repeats = 1 if params.pruner['mode'] != 'random_weighted'\
                        else int(params.pruner['repeats'])
            for i in range(repeats): 
                model = origModel
                params.pruner['pruning_perc'] = pp
                pruner = setup_pruners(params, model)
                pn = i if params.pruner['mode'] == 'random_weighted' else None
                channelsPruned, prunedModel = pruner.prune_model(model, pruneNum=pn)
#}}}
