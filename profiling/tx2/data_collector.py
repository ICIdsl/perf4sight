import os
import sys
import copy
import json
import time
import random 
import itertools
import subprocess
import configparser as cp 

import torch 
import torch.cuda
import numpy as np
import pandas as pd
import torch.nn as nn

import utils
import profiling.tx2.run_test as test_runner
from profiling.tx2.cuda_mem_prof import MemoryProfiler

class DataCollector(object): 
    def __init__(self, params):
    #{{{
        self.globalParams = params

        self.config = cp.ConfigParser()
        self.logDir = self.globalParams.profile['log_dir']
        self.save = eval(self.globalParams.profile['save'])
        self.config.read(self.globalParams.profile['test_config'])

        self.tests = []
    #}}}    

    def get_test_name(self, sec):
        return test_runner.get_test_from_name(sec)

    def collect(self, gpu): 
    #{{{
        """ Run through all sections in config file and collect requested data """
        sections = self.config.sections()
        assert len(sections) != 0, "config file reading issue, it has no sections"
        self.create_tests(sections)
        self.run_tests(gpu)
    #}}}

    def create_tests(self, sections): 
    #{{{
        for sec in sections: 
            test = self.get_test_name(sec)
            config = copy.deepcopy(dict(self.config[sec].items()))
            config = {k:eval(v) for k,v in config.items()}
            tests = list(itertools.product(*list(config.values())))
            keys = list(config.keys())
            params = [{k:v[i] for i,k in enumerate(keys)} for v in tests]
            self.tests = [{sec: copy.deepcopy(p)} for p in params if test.parse(p)] 
    #}}}
    
    def log(self, results):
    #{{{
        def preprocess_mem_results(): 
        #{{{
            if results['memprof'] is not None:
                memAllocated = np.mean(results['memprof']['max_alloc'])
                memReserved = np.mean(results['memprof']['max_reserved'])
                memUsed = np.mean(results['memprof']['max_used'])
                memAvail = np.mean(results['memprof']['max_avail'])
                memFree = np.mean(results['memprof']['free'])
                totalCapacity = np.mean(results['memprof']['capacity'])

                if eval(results['success']):
                    print("Test SUCCESS!\nMemory Used: {:.2f}, Memory Available: {:.2f}".format(memUsed, memAvail))
                else:
                    print("Memory Used: {:.2f}, Memory Available: {:.2f}".format(memUsed, memAvail))

                return [memAllocated, memReserved, memUsed, memAvail, memFree, totalCapacity]
            
            return [-1,-1,-1,-1,-1]
        #}}}

        def save_mem_results(memResults):
        #{{{
            if self.save:
                logCsvFile = results['params']['model'].split('/')[-1]
                logCsvFile = "{}.csv".format(logCsvFile.split('.')[0])
                
                logDir = os.path.join(self.logDir, 'memory')
                cmd = f"mkdir -p {logDir}"
                subprocess.run(cmd, shell=True, check=True)
                
                logCsv = "{}/{}".format(logDir, logCsvFile)
                
                values = [results['success'], *memResults, results['params']]
                values = [str(x) for x in values]

                if logCsv not in self.logs:
                    self.logs.append(logCsv)
                
                # check if file exists
                if not os.path.isfile(logCsv): 
                    with open(logCsv, 'w') as f: 
                        f.write('passed|mem_alloc|mem_reserved|mem_used|mem_avail|free|capacity|params\n')
                    
                with open(logCsv, 'a') as f: 
                    f.write('|'.join(values)+'\n')
        #}}}

        def preprocess_latency_results(): 
        #{{{
            if results['times'] is not None:
                latency = np.mean(results['times'])
                if eval(results['success']): 
                    print("Test SUCCESS!\nLatency: {:.2f}".format(latency))
                return latency
        #}}}
        
        def save_latency_results(latency):
        #{{{
            if self.save:
                logCsvFile = results['params']['model'].split('/')[-1]
                logCsvFile = "{}.csv".format(logCsvFile.split('.')[0])
                
                logDir = os.path.join(self.logDir, 'latency')
                cmd = f"mkdir -p {logDir}"
                subprocess.run(cmd, shell=True, check=True)

                logCsv = "{}/{}".format(logDir, logCsvFile)
                
                values = [results['success'], latency, results['params']]
                values = [str(x) for x in values]

                if logCsv not in self.logs:
                    self.logs.append(logCsv)
                
                # check if file exists
                if not os.path.isfile(logCsv): 
                    with open(logCsv, 'w') as f: 
                        f.write('passed|latency|params\n')
                    
                with open(logCsv, 'a') as f: 
                    f.write('|'.join(values)+'\n')
        #}}}

        if results['params']['test_var'] == 'memory':
            memData = preprocess_mem_results()
            save_mem_results(memData)
        else:
            latency = preprocess_latency_results()
            save_latency_results(latency)
    #}}}

    def run_tests(self, gpu):
    #{{{
        def allocate_idle_mem():
        #{{{
            a = None
            if eval(self.globalParams.profile['limit_mem']):
                idleMem = eval(self.globalParams.profile['idle_mem'])
                print(f"Adding idle memory of {idleMem}GB")
                idleParams = int((float(idleMem)*1024*1024*1024)/4)
                a = torch.randn(size=(idleParams,))
            return a 
        #}}}

        def cleanup():
        #{{{
            print("Running cleanup")
            expandedLogs = [utils.expand_log_headers(pd.read_csv(log, delimiter='|'))[1] for log in self.logs]
            [eLog.to_csv(f.split('.')[0]+"-gt.csv", sep='|', index=False) for (f,eLog) in zip(self.logs, expandedLogs)]
        #}}}
        
        try:
            testNum = 1
            self.logs = []
            nets2Ignore = []
            for test in self.tests:
            #{{{
                if list(test.values())[0]['model'] in nets2Ignore: 
                    if eval(self.globalParams.profile['till_first_fail']): 
                        testNum += 1
                        continue
                    else:
                        MemoryProfiler.enabled=True
                        idleAvail = MemoryProfiler.get_mem_avail()
                        stats = {'max_alloc': -1, 'max_reserved': -1, 'max_used': -1,\
                                 'max_avail': idleAvail, 'free': -1, 'capacity': -1}
                        params = [*test.values()][0]
                        testVar = params['test_var']
                        if testVar == 'memory':
                            results = {'success': str(False), 'params': params, 'memprof': stats}
                        else:
                            results = {'success': str(False), 'params': params, 'times': None}
                else:
                #{{{
                    start = time.time()
                    print("\n==> Running Test {}/{}".format(testNum, len(self.tests)))

                    root = os.path.dirname(__file__)
                    tmpFile = os.path.join(root, 'tmp/test_{}_{}.json'.format(list(test.keys())[0], gpu))
                    test['gpu'] = int(gpu)

                    cmd = f'mkdir -p {root}/tmp'
                    subprocess.check_call(cmd.split(' '))
                    with open(tmpFile, 'w') as f: 
                        json.dump(test, f)
            
                    executable = os.path.join(root, 'run_test.py')
                    cmd = 'python {} --params {}'.format(executable, tmpFile)
                    try:
                        print(f"Called profiling function")
                        subprocess.check_call(cmd.split(' '))
                        with open(tmpFile) as f:
                            results = json.load(f)
                    except Exception as e:
                        torch.cuda.empty_cache()
                        print("Test FAILED: {}".format(e))
                        with open(tmpFile, 'r') as f: 
                            stats = json.load(f)
                        params = [*test.values()][0]
                        testVar = params['test_var']
                        if testVar == 'memory':
                            if stats['max_used'][0] != 0:
                                stats['max_used'] = [x - stats['max_avail'][0] for x in stats['max_used']]
                            results = {'success': str(False), 'params': params, '{}'.format('memprof' if testVar=='memory' else 'times'): stats}
                        else:
                            results = {'success': str(False), 'params': params, '{}'.format('memprof' if testVar=='memory' else 'times'): None}
                    
                    print("Time taken: {:.2f}".format(time.time() - start))
                #}}}
                
                torch.cuda.empty_cache()
                self.log(results)
                testNum += 1
                
                # if eval(self.globalParams.profile['till_first_fail']): 
                if not eval(results['success']):
                    net = list(test.values())[0]['model']
                    nets2Ignore.append(net)
            #}}}
                    
            cleanup()
        except Exception as e:
            print("Test FAILED (outer loop): {}".format(e))
            # convert to expanded header format
            cleanup()
    #}}}
