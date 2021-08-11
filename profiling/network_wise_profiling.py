import os
import sys
import copy
import json
import glob
import time
import random
import argparse
import subprocess
import configparser as cp

import torch
import torch.cuda
import torch.nn as nn

import math
import numpy as np
import pandas as pd

import utils

def perform_profiling(params):
#{{{
    if params.profile['device'] == 'tx2':
        from profiling.tx2.data_collector import DataCollector 
    elif params.profile['device'] == '2080ti':
        from profiling.ti2080.data_collector import DataCollector 

    dataCol = DataCollector(params)
    dataCol.collect(gpu= params.profile['gpu_id'])
#}}}

def create_profiling_config(params):
#{{{
    _params = params.profile
    currdir = '/'.join(__file__.split('/')[:-1])
    storedir = os.path.join(currdir, 'profiling_configs')
    utils.create_dir(storedir)
    configfile = os.path.join(storedir, f"{_params['network']}.ini")

    with open(configfile, 'w') as f:
        print(f"[{_params['network'].upper()}]", file=f)
        print(f"test_var= {eval(_params['profile_variables'])}", file=f)
        print(f"bs= {eval(_params['bs'])}", file=f)
        print(f"dataset= ['{_params['dataset']}']", file=f)
        print(f"net_name= ['{_params['network']}']", file=f)
        print(f"exec_stage= {eval(_params['stage'])}", file=f)

        allModels = glob.glob(os.path.join(_params['local_model_loc'], '*.py'))
        models = [x for x in allModels if _params['network'] in x]
        modelList = 'model= ['
        offset = len(modelList)
        for i,m in enumerate(models):
            prefix = '' if i == 0 else ' '*offset
            body = f"'{m.replace(os.path.dirname(m), _params['device_model_loc'])}'"
            suffix = ',\n' if i+1 < len(models) else ']'
            modelList += prefix + body + suffix
        print(modelList, file=f)

    print(f"Copy config created at {configfile} to device")
#}}}

def profile_network(params):
#{{{
    if eval(params.profile['create_config']):
        print(f"Creating profiling configs...")
        create_profiling_config(params)

    if eval(params.profile['run_profiling']):
        print(f"Profiling networks on device {params.profile['device']}....")
        perform_profiling(params)
#}}}
