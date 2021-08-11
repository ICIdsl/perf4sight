import os
import sys
import copy
import importlib
import subprocess
import pandas as pd

import torch
import torch.nn as nn

def create_dir(saveDir): 
#{{{
    if not os.path.isdir(saveDir): 
        cmd = f"mkdir -p {saveDir}"
        subprocess.run(cmd, check=True, shell=True)
#}}}

def expand_log_headers(log): 
#{{{
    """ Takes a dataframe as input with 'Params' being a dictionary
    It expands the keys the in the dict into headers into columns 
    in the dataframe """
    
    currHeaders = log.columns.tolist()
    currHeaders.remove('params')
    expandedLog = {}
    for key in currHeaders: 
        expandedLog[key] = log[key].tolist()  
    params = [eval(x) for x in log.params.tolist()]
    paramHeaders = list(eval(log.params[0]).keys())
    for key in paramHeaders: 
        expandedLog[key] = [x[key] if type(x[key]) != list else x[key][0] for x in params]

    return paramHeaders, pd.DataFrame(expandedLog) 
#}}}

def read_model(modelFile, netName, depth=None):
#{{{
    """Takes path to .py model description file and name of model function
        and returns the model"""
    sys.path.append('/')
    modelFile = modelFile.split('.')[0]
    path = modelFile.replace('/','.')[1:]
    module = importlib.import_module(path)
    model = module.__dict__[netName]() if depth is None else module.__dict__[netName](depth=depth)
    return model
#}}}

class TestGenerator(object): 
#{{{
    def __init__(self): 
        self.layerCount = 0
        self.inputSizes = {}
        
    def forward_hook(self, module, input, output): 
        self.inputSizes[self.layerCount] = input[0].shape 
        self.layerCount += 1

    def from_model(self, model, dataset, modelConfig): 
    #{{{
        """ Generates a layer by layer test given an input model by identifying conv and fc layers """
        hooks = []
        for n,m in model.named_modules(): 
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(self.forward_hook))
        
        batchSize = 8
        ipImgSize = [32,32] if 'cifar' in dataset else [224,224]
        dummyIp = torch.FloatTensor(batchSize, 3, *ipImgSize)
        
        self.layerCount = 0
        model(dummyIp)        
        
        [h.remove() for h in hooks]

        configOpFile = modelConfig 
        allTests = []
        self.layerCount = 0
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                keys = {'ic':'in_channels', 'oc':'out_channels', 'k':'kernel_size', 'pad':'padding', 'stride':'stride', 'groups':'groups'}
                test = ["[BASIC_CONV-{}]".format(n)]  
                test.append("mode=single")
                test.append("bs={}".format(batchSize))
                test.append("bias={}".format(m.bias is not None))
                ipSize = list(self.inputSizes[self.layerCount])[2:]
                test.append("ip={}".format(ipSize))
                for k,v in keys.items():
                    test.append("{}={}".format(k,eval("m.{}".format(v))))
                allTests.append('\n'.join(test))

                self.layerCount += 1
           
        dirName = os.path.dirname(configOpFile)
        cmd = f"mkdir -p {dirName}"
        subprocess.check_call(cmd, shell=True)
        with open(configOpFile, 'w') as f:
            f.write('\n\n'.join(allTests))
    #}}}
#}}}
