import os
import subprocess

import torch

from pynvml import *

class HookMemoryProfiler(): 
    enabled = False

    def __init__(self, description, model, log=None, verbose=False, gpu=0): 
    #{{{
        self.desc = description
        self.log = None
        self.data = None
        self.verbose = verbose
        self.gpu = int(gpu)
        
        self.model = model
        
        nvmlInit()
        self.gpuHandle = nvmlDeviceGetHandleByIndex(self.gpu)

        if self.enabled :
            self.data = {'max_alloc':[], 'max_reserved':[], 'max_used':[], 'max_avail':[], 'capacity':[]}

            if log is not None:
                self.log = log
                self.headers = ['batch_size', 'max_alloc', 'max_reserved', 'max_used']
                if not os.path.isfile(self.log): 
                    with open(self.log, 'w') as logFile: 
                        logFile.write(','.join(self.headers)+'\n')
    #}}}

    def memory_hook(self, *args):
        memUsed =  nvmlDeviceGetMemoryInfo(self.gpuHandle).used * 2**-20
        if memUsed > self.maxMemUsed: 
            self.maxMemUsed = memUsed

    @classmethod
    def disable(cls):
        cls.enabled = False

    def reset(self): 
        self.hooks = []
        self.maxMemUsed = 0

    def __enter__(self): 
    #{{{
        def func(): 
            torch.cuda.reset_peak_memory_stats(device=self.gpu)
            torch.cuda.empty_cache()

            self.data['max_avail'] = nvmlDeviceGetMemoryInfo(self.gpuHandle).free * 2**-20
            
            self.maxMemUsed = 0
            self.hooks = []
            for mod in self.model.modules(): 
                self.hooks.append(mod.register_forward_hook(self.memory_hook))
                self.hooks.append(mod.register_backward_hook(self.memory_hook))
            
        if self.enabled: 
            func()
    #}}}

    def __exit__(self, type, value, traceback): 
    #{{{
        def func(): 
            maxAllocated = torch.cuda.max_memory_allocated(device=self.gpu) * 2**-20
            maxReserved = torch.cuda.max_memory_reserved(device=self.gpu) * 2**-20
            maxTotal = nvmlDeviceGetMemoryInfo(self.gpuHandle).total * 2**-20
            
            if self.verbose:
                print("==> Stats for Memory Profiler - {} (After) ".format(self.desc))
                print("\tMax Memory Allocated = {:.2f}MB".format(maxAllocated))
                print("\tMax Memory Reserved = {:.2f}MB".format(maxReserved))
                print("\tMax Memory Used = {:.2f}MB".format(self.maxMemUsed))
                print("\tCUDA-overhead = {:.2f}MB".format(self.maxMemUsed - maxReserved))
                print("\tPyTorch-overhead = {:.2f}MB".format(maxReserved - maxAllocated))
                
            self.data['max_alloc'].append(maxAllocated)
            self.data['max_reserved'].append(maxReserved)
            self.data['max_used'].append(self.maxMemUsed)
            self.data['capacity'].append(maxTotal)

            [h.remove() for h in self.hooks]
                
        if self.enabled: 
            func()
    #}}}
