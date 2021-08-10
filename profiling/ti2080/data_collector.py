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
import numpy as np
import pandas as pd
import torch.nn as nn

import utils
import profiling.ti2080.run_test as test_runner
from profiling.tx2.data_collector import DataCollector as BaseDataCollector

class DataCollector(BaseDataCollector): 
    def get_test_name(self, sec):
        return test_runner.get_test_from_name(sec)
    
    def log(self, results):
    #{{{
        def preprocess_mem_results(): 
        #{{{
            if results['memprof'] is not None:
                memAllocated = np.mean(results['memprof']['max_alloc'])
                memReserved = np.mean(results['memprof']['max_reserved'])
                memUsed = np.mean(results['memprof']['max_used'])
                memAvail = np.mean(results['memprof']['max_avail'])
                totalCapacity = np.mean(results['memprof']['capacity'])
                
                pytorchOverhead = memReserved - memAllocated 
                cudaOverhead = memUsed - memReserved
                if eval(results['success']):
                    print("Test SUCCESS!\nMemory Used: {:.2f}, Predicted PyTorch Overead: {:.2f}, Predicted CUDA Overhead: {:.2f}".format(memUsed, pytorchOverhead, cudaOverhead))
                else:
                    print("Memory Used: {:.2f}, Predicted PyTorch Overead: {:.2f}, Predicted CUDA Overhead: {:.2f}".format(memUsed, pytorchOverhead, cudaOverhead))

                return [memAllocated, memReserved, memUsed, memAvail, totalCapacity]
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
                
                # values = [results['success'], memAllocated, memReserved, memUsed, memAvail, totalCapacity, results['params']]
                values = [results['success'], *memResults, results['params']]
                values = [str(x) for x in values]

                if logCsv not in self.logs:
                    self.logs.append(logCsv)
                
                # check if file exists
                if not os.path.isfile(logCsv): 
                    with open(logCsv, 'w') as f: 
                        f.write('passed|mem_alloc|mem_reserved|mem_used|mem_avail|capacity|params\n')
                    
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
        testNum = 1
        self.logs = []
        for test in self.tests:
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
            subprocess.run(cmd, shell=True, check=True)

            with open(tmpFile) as f:
                results = json.load(f)
            
            self.log(results)
            testNum += 1
            print("Time taken: {:.2f}".format(time.time() - start))
        
        # convert to expanded header format
        expandedLogs = [utils.expand_log_headers(pd.read_csv(log, delimiter='|'))[1] for log in self.logs]
        [eLog.to_csv(f.split('.')[0]+"-gt.csv", sep='|', index=False) for (f,eLog) in zip(self.logs, expandedLogs)]
    #}}}
