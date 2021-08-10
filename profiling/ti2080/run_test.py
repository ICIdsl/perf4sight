import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from pynvml import *

import os
import sys
import math
import copy
import json
import argparse
from abc import ABC, abstractmethod

p4sPath = []
for direc in __file__.split('/'):
    p4sPath.append(direc)
    if direc == 'perf4sight':
        break
p4sPath = '/'.join(p4sPath)
sys.path.append(p4sPath)

import utils
from profiling.ti2080.cuda_mem_prof import HookMemoryProfiler

class BasicTest(ABC): 
#{{{
    testType = None
    
    def __init__(self, name, params, verbose=False): 
        self.name = name
        self.params = copy.deepcopy(params)
        self.verbose = verbose
        self.success = False

    @classmethod
    @abstractmethod
    def parse(params):
        pass

    @abstractmethod
    def create_dummy_input(self):
        pass

    @abstractmethod
    def create_layer(self):
        pass

    def run(self, gpu, memTest=True, model=None): 
    #{{{
        print("{}: {}".format(self.name, self.params))
        self.gpu = gpu
        try:
            if self.params['test_var'] == 'memory':
                print("Running Memory Test")
                self.run_memory_test_gpu()
            else:
                print("Running Latency Test")
                self.run_latency_test_gpu()
            self.success = True
        except (RuntimeError, Exception) as e:
            print("Test FAILED: {}".format(e))
    #}}}

    def run_memory_test_gpu(self): 
    #{{{
        def train(): 
        #{{{
            output = self.model(inputTensor)
            loss = criterion(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        #}}}
        
        if self.verbose:
            print("========== Running {}".format(self.name))
            print(self.params)

        inputTensor = self.create_dummy_input()
        self.create_layer()
        modelSize = sum([np.prod(x.shape) for x in self.model.parameters()]) * 4 * 2**-20
        print("Model Size = {:.2f}MB".format(modelSize))
        with torch.cuda.device(self.gpu): 
            [x.pin_memory() for x in self.model.parameters()]
            self.model = nn.DataParallel(self.model, [self.gpu])
            self.model.to(f"cuda:{self.gpu}", non_blocking=True)
            inputTensor = inputTensor.cuda(self.gpu)
            numClasses = 10 if self.params['dataset'] == 'cifar10' else 100 if self.params['dataset'] == 'cifar100' else 1000
            targets = torch.empty(self.params['bs'], dtype=torch.long).random_(numClasses).cuda(self.gpu)
            
            criterion = torch.nn.CrossEntropyLoss()
            optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            
            self.model.train()
            
            # self.gpu is the absolute gpu which is required for smi query in memory profiler
            HookMemoryProfiler.enabled = True
            self.memProfiler = HookMemoryProfiler("Single Test", self.model, log=None, gpu=self.gpu, verbose=False)
            for i in range(2):
                with self.memProfiler:
                    train()
        
        if self.verbose:
            print(self.memProfiler.data)
    #}}}
    
    def run_latency_test_gpu(self): 
    #{{{
        def train(): 
        #{{{
            output = self.model(inputTensor)
            loss = criterion(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        #}}}
        
        if self.verbose:
            print("========== Running {}".format(self.name))
            print(self.params)

        inputTensor = self.create_dummy_input()
        self.create_layer()
        modelSize = sum([np.prod(x.shape) for x in self.model.parameters()]) * 4 * 2**-20
        print("Model Size = {:.2f}MB".format(modelSize))
        with torch.cuda.device(self.gpu): 
            [x.pin_memory() for x in self.model.parameters()]
            self.model = nn.DataParallel(self.model, [self.gpu])
            self.model.to(self.gpu, non_blocking=True)
            inputTensor = inputTensor.cuda(self.gpu)
            numClasses = 10 if self.params['dataset'] == 'cifar10' else 100 if self.params['dataset'] == 'cifar100' else 1000
            targets = torch.empty(self.params['bs'], dtype=torch.long).random_(numClasses).cuda(self.gpu)
            
            criterion = torch.nn.CrossEntropyLoss()
            optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            
            self.model.train()
            self.timingData = []
            
            # warmup run
            train()

            start = torch.cuda.Event(enable_timing=True)            
            end = torch.cuda.Event(enable_timing=True)            
            # self.gpu is the absolute gpu which is required for smi query in memory profiler
            for i in range(3):
                start.record()
                train()
                end.record()
                torch.cuda.synchronize()
                self.timingData.append(start.elapsed_time(end))
        
        if self.verbose:
            print(self.memProfiler.data)
    #}}}
#}}}

class AlexNet(BasicTest): 
#{{{
    testType = 'AlexNet'

    def parse(params):
    #{{{
        assertMsg = "Required parameters are : bs, dataset, net_name, model"
        paramsCorrect = all([x in params.keys() for x in ['bs', 'dataset', 'net_name', 'model']])
        assert paramsCorrect, assertMsg

        return True        
    #}}}
    
    def create_dummy_input(self):
    #{{{
        ip = (32,32) if 'cifar' in self.params['dataset'] else (224,224)
        return torch.FloatTensor(self.params['bs'], 3, *ip)
    #}}}

    def create_layer(self):
    #{{{
        netName = self.params['net_name']
        modelFile = self.params['model']
        self.model = utils.read_model(modelFile, netName)
    #}}}
#}}}

class ResNet(BasicTest): 
#{{{
    testType = 'ResNet'

    def parse(params):
    #{{{
        assertMsg = "Required parameters are : bs, dataset, net_name, model"
        paramsCorrect = all([x in params.keys() for x in ['bs', 'dataset', 'net_name', 'model']])
        assert paramsCorrect, assertMsg

        return True        
    #}}}
    
    def create_dummy_input(self):
    #{{{
        ip = (32,32) if 'cifar' in self.params['dataset'] else (224,224)
        return torch.FloatTensor(self.params['bs'], 3, *ip)
    #}}}

    def create_layer(self):
    #{{{
        netName = self.params['net_name']
        modelFile = self.params['model']
        self.model = utils.read_model(modelFile, netName)
    #}}}
#}}}

class MobileNetV2(BasicTest): 
#{{{
    testType = 'MobileNetV2'

    def parse(params):
    #{{{
        assertMsg = "Required parameters are : bs, dataset, net_name, model"
        paramsCorrect = all([x in params.keys() for x in ['bs', 'dataset', 'net_name', 'model']])
        assert paramsCorrect, assertMsg

        return True        
    #}}}
    
    def create_dummy_input(self):
    #{{{
        ip = (32,32) if 'cifar' in self.params['dataset'] else (224,224)
        return torch.FloatTensor(self.params['bs'], 3, *ip)
    #}}}

    def create_layer(self):
    #{{{
        netName = self.params['net_name']
        modelFile = self.params['model']
        self.model = utils.read_model(modelFile, netName)
    #}}}
#}}}

class SqueezeNet(BasicTest): 
#{{{
    testType = 'SqueezeNet'

    def parse(params):
    #{{{
        assertMsg = "Required parameters are : bs, dataset, net_name, model"
        paramsCorrect = all([x in params.keys() for x in ['bs', 'dataset', 'net_name', 'model']])
        assert paramsCorrect, assertMsg

        return True        
    #}}}
    
    def create_dummy_input(self):
    #{{{
        ip = (32,32) if 'cifar' in self.params['dataset'] else (224,224)
        return torch.FloatTensor(self.params['bs'], 3, *ip)
    #}}}

    def create_layer(self):
    #{{{
        netName = self.params['net_name']
        modelFile = self.params['model']
        self.model = utils.read_model(modelFile, netName)
    #}}}
#}}}

def get_test_from_name(name): 
#{{{
    if 'ALEXNET' in name:
        test = AlexNet
    elif 'RESNET' in name:
        test = ResNet 
    elif 'MOBILENET' in name:
        test = MobileNetV2 
    elif 'SQUEEZENET' in name:
        test = SqueezeNet 
    else:
        raise ValueError('Test not specified for type - {}'.format(name))
    
    return test
#}}}

def run_test(params):   
#{{{
    gpu = params.pop('gpu')
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(): 
        assert str(gpu) in os.environ['CUDA_VISIBLE_DEVICES'].split(','), 'specified gpu {} is not in CUDA_VISIBLE_DEVICES - {}'.format(gpu, os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        print("ERROR : set environment variable CUDA_VISIBLE_DEVICES before running this")
        sys.exit()

    testName = list(params.keys())[0] 
    testParams = list(params.values())[0]
    
    test = get_test_from_name(testName)

    testObj = test(testName, testParams)
    testObj.run(gpu)
    return testObj
#}}}

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run single test")
    parser.add_argument('--params', type=str, help='path to file with json of params')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.params) as jsonF:
        params = json.load(jsonF)
    
    test = run_test(params)
     
    if test.params['test_var'] == 'memory':
        results = {'success': str(test.success), 'test_type': test.testType, 'test_name': test.name, 'params': test.params, 'memprof': test.memProfiler.data}
    else:
        results = {'success': str(test.success), 'test_type': test.testType, 'test_name': test.name, 'params': test.params, 'times': test.timingData}

    with open(args.params, 'w') as jsonF:
        json.dump(results, jsonF)

if __name__ == '__main__':
    main()
    
