import os
import sys
import json
import subprocess

import torch

class MemoryProfiler(): 
    enabled = False

    def __init__(self, description, model, log=None, verbose=False, gpu=0): 
    #{{{
        self.desc = description
        self.log = None
        self.data = None
        self.verbose = verbose
        self.gpu = int(gpu)
        
        self.model = model
        
        if self.enabled :
            self.data = {'max_alloc':[], 'max_reserved':[], 'max_used':[], 'max_avail':[], 'free':[], 'capacity':[]}
            self.log = log
            self.completed = False
    #}}}

    def memory_hook(self, *args):
    #{{{
        try: 
            memUsed, _, memFree = self.get_mem_usage() 
            if memUsed > self.maxMemUsed: 
                self.maxMemUsed = memUsed
            if self.log is not None and not self.completed:
                self.data['max_alloc'] = [torch.cuda.max_memory_allocated() * 2**-20]
                self.data['max_reserved'] = [torch.cuda.max_memory_reserved() * 2**-20]
                self.data['max_used'] = [self.maxMemUsed]
                with open(self.log, 'w') as f: 
                    json.dump(self.data, f)
                     
        except Exception as e: 
            print("mem_hook: Cannot read GPU memory - {}".format(e))
            raise e
    #}}}

    @classmethod
    def disable(cls):
        cls.enabled = False

    @classmethod 
    def get_mem_avail(cls): 
    #{{{
        with open('/proc/meminfo', 'r') as f: 
            f.readline()
            f.readline()
            memAvail = int(f.readline().strip().split(':')[-1].strip().split(' ')[0])
        
        return memAvail * 2**-10
    #}}}

    @classmethod
    def get_mem_usage(cls): 
    #{{{
        with open('/proc/meminfo', 'r') as f: 
            memTotal = int(f.readline().strip().split(':')[-1].strip().split(' ')[0])
            memFree = int(f.readline().strip().split(':')[-1].strip().split(' ')[0])
            f.readline()
            memBuff = int(f.readline().strip().split(':')[-1].strip().split(' ')[0])
            memCache = int(f.readline().strip().split(':')[-1].strip().split(' ')[0]) 
        
        used = (memTotal - memFree - memBuff - memCache) * 2**-10
        tot = memTotal * 2**-10

        return (int(used), int(tot), int(memFree * 2**-10))
    #}}}
    
    def __enter__(self): 
    #{{{
        def func(): 
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
                
            self.maxMemUsed = 0
            self.hooks = []
            for mod in self.model.modules(): 
                self.hooks.append(mod.register_forward_hook(self.memory_hook))
                self.hooks.append(mod.register_backward_hook(self.memory_hook))

            return self
            
        if self.enabled: 
            func()
    #}}}

    def __exit__(self, type, value, traceback): 
    #{{{
        def func(): 
            maxAllocated = torch.cuda.max_memory_allocated() * 2**-20
            maxReserved = torch.cuda.max_memory_reserved() * 2**-20
            
            if self.verbose:
                print("==> Stats for Memory Profiler - {} (After) ".format(self.desc))
                print("\tMax Memory Allocated = {:.2f}MB".format(maxAllocated))
                print("\tMax Memory Reserved = {:.2f}MB".format(maxReserved))
                print("\tMax Memory Used = {:.2f}MB".format(self.maxMemUsed))
                
            self.data['max_alloc'].append(maxAllocated)
            self.data['max_reserved'].append(maxReserved)
            self.data['max_used'].append(self.maxMemUsed)

            self.completed = True

            [h.remove() for h in self.hooks]
                
        if self.enabled: 
            func()
    #}}}

