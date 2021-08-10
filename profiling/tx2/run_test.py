import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import os
import sys
import math
import time
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
from profiling.tx2.cuda_mem_prof import MemoryProfiler

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

    def get_single_layer(self):
        if self.params.pop('mode') == 'single':
            self.parse_test()
            return (self.create_dummy_input(), self.create_layer())

    def run(self, gpu, logFile, memTest=True, model=None): 
    #{{{
        print("{}: {}".format(self.name, self.params))
        self.gpu = gpu
        self.logFile = logFile
        if self.params['test_var'] == 'memory':
            print("Running Memory Test")
            self.run_memory_test_gpu()
        elif self.params['test_var'] == 'latency':
            print("Running GPU Latency Test")
            self.run_latency_test_gpu()
        elif self.params['test_var'] == 'cpu-latency':
            print("Running CPU Latency Test")
            self.run_latency_test_cpu()
        else:
            raise ValueError(f"Test for {self.params['test_var']} is not defined")
    #}}}

    def run_memory_test_gpu(self): 
    #{{{
        def inference():
            with torch.no_grad():
                self.model(inputTensor)

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
        
        MemoryProfiler.enabled = True
        idleMemUsed, capacity, idleFree = MemoryProfiler.get_mem_usage()
        idleAvail = MemoryProfiler.get_mem_avail()

        inputTensor = self.create_dummy_input()
        self.create_layer()
        modelSize = sum([np.prod(x.shape) for x in self.model.parameters()]) * 4 * 2**-20
        print("Model Size = {:.2f}MB".format(modelSize))
        
        with torch.cuda.device(0): 
            [x.pin_memory() for x in self.model.parameters()]
            self.model = nn.DataParallel(self.model, [0])
            self.model.to(0, non_blocking=True)
            inputTensor = inputTensor.cuda(0)
            
            if self.params['exec_stage'] == 'inference':
                self.model.eval() 
            else:
                numClasses = 10 if self.params['dataset'] == 'cifar10' else\
                        100 if self.params['dataset'] == 'cifar100' else 1000
                targets = torch.empty(self.params['bs'], dtype=torch.long).random_(numClasses).cuda(0)
                criterion = torch.nn.CrossEntropyLoss()
                optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                self.model.train()
            
            self.memProfiler =\
                    MemoryProfiler("Single Test", self.model, log=self.logFile, gpu=self.gpu, verbose=False)
            self.memProfiler.data['max_alloc'] = [0]
            self.memProfiler.data['max_reserved'] = [0]
            self.memProfiler.data['max_used'] = [0]
            self.memProfiler.data['max_avail'].append(idleAvail)
            self.memProfiler.data['capacity'].append(capacity) 
            self.memProfiler.data['free'].append(idleFree)
            with open(self.memProfiler.log, 'w') as f: 
                json.dump(self.memProfiler.data, f)
            
            try:
                with self.memProfiler:
                    print("Warmup run")
                    train() if self.params['exec_stage'] == 'training' else inference()
            except Exception as e:
                self.success = False  
                print("Warmup Run FAILED: {}".format(e))
                return
            
            try:
                idleMems = []
                for i in range(3):
                    with self.memProfiler:
                        train() if self.params['exec_stage'] == 'training' else inference()
                        self.success = True
                self.memProfiler.data['max_used'] = [x-idleMemUsed for x in self.memProfiler.data['max_used']]
            except Exception as e:
                self.success = False
                print("Test FAILED: {}".format(e))
            
                
        if self.verbose:
            print(self.memProfiler.data)
    #}}}
    
    def run_latency_test_gpu(self): 
    #{{{
        def inference():
            with torch.no_grad():
                output = self.model(inputTensor)
        
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
        with torch.cuda.device(0): 
            [x.pin_memory() for x in self.model.parameters()]
            self.model = nn.DataParallel(self.model, [0])
            self.model.to(0, non_blocking=True)
            inputTensor = inputTensor.cuda(0)
            
            if self.params['exec_stage'] == 'inference':
                self.model.eval() 
            else:
                numClasses = 10 if self.params['dataset'] == 'cifar10' else\
                        100 if self.params['dataset'] == 'cifar100' else 1000
                targets = torch.empty(self.params['bs'], dtype=torch.long).random_(numClasses).cuda(0)
                criterion = torch.nn.CrossEntropyLoss()
                optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                self.model.train()
            
            self.timingData = []
            #### warmup run
            try:
                train() if self.params['exec_stage'] == 'training' else inference()
            except Exception as e:
                self.success = False
                print("Test FAILED: {}".format(e))
                return
            
            start = torch.cuda.Event(enable_timing=True)            
            end = torch.cuda.Event(enable_timing=True)            
            for i in range(3):
                try:
                    start.record()
                    train() if self.params['exec_stage'] == 'training' else inference()
                    end.record()
                    torch.cuda.synchronize()
                    self.timingData.append(start.elapsed_time(end))
                    self.success = True
                except Exception as e:
                    self.success = False
                    print("Test FAILED: {}".format(e))
        
        if self.verbose:
            print(self.memProfiler.data)
    #}}}
    
    def run_latency_test_cpu(self): 
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
        numClasses = 10 if self.params['dataset'] == 'cifar10' else 100 if self.params['dataset'] == 'cifar100' else 1000
        targets = torch.empty(self.params['bs'], dtype=torch.long).random_(numClasses)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # for p in self.model.parameters():
        #     p.requires_grad = False
        # for n,m in self.model.named_modules():
        #     if isinstance(m, nn.Linear):
        #         for p in m.parameters(): 
        #             p.requires_grad = True
        
        self.model.eval()
        
        self.timingData = []
        #### warmup run
        try:
            train()
        except Exception as e:
            self.success = False
            print("Test FAILED: {}".format(e))
            return
        
        for i in range(3):
            try:
                start = time.time()
                train()
                end = time.time()
                self.timingData.append((end-start) * 1000)
                self.success = True
            except Exception as e:
                self.success = False
                print("Test FAILED: {}".format(e))
        
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

class VGG(BasicTest): 
#{{{
    testType = 'VGG'

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

class MnasNet(BasicTest): 
#{{{
    testType = 'MnasNet'

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

class GoogLeNet(BasicTest): 
#{{{
    testType = 'GoogLeNet'

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
    elif 'OFA_RESNET' in name:
        test = ResNet 
    elif 'GATED_RESNET' in name:
        test = GatedResNet 
    elif 'RESNET' in name:
        test = ResNet 
    elif 'MOBILENET' in name:
        test = MobileNetV2 
    elif 'SQUEEZENET' in name:
        test = SqueezeNet 
    elif 'VGG' in name:
        test = VGG
    elif 'MNASNET' in name:
        test = MnasNet
    elif 'GOOGLENET' in name:
        test = GoogLeNet
    else:
        raise ValueError('Test not specified for type - {}'.format(name))
    
    return test
#}}}

def run_test(logFile, params):   
#{{{
    gpu = params.pop('gpu')
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(): 
        assert str(gpu) in os.environ['CUDA_VISIBLE_DEVICES'].split(','),\
         'specified gpu {} is not in CUDA_VISIBLE_DEVICES - {}'.format(gpu, os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        print("ERROR : set environment variable CUDA_VISIBLE_DEVICES before running this")
        sys.exit()

    testName = list(params.keys())[0] 
    testParams = list(params.values())[0]
    
    test = get_test_from_name(testName)

    testObj = test(testName, testParams)
    testObj.run(gpu, logFile)
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
    
    test = run_test(args.params, params)
    torch.cuda.empty_cache()
     
    if test.params['test_var'] == 'memory':
        results = {'success': str(test.success), 'test_type': test.testType, 'test_name': test.name, 'params': test.params, 'memprof': test.memProfiler.data}
    else:
        results = {'success': str(test.success), 'test_type': test.testType, 'test_name': test.name, 'params': test.params, 'times': test.timingData}

    with open(args.params, 'w') as jsonF:
        json.dump(results, jsonF)

if __name__ == '__main__':
    main()
    
