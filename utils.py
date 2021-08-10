import os
import sys
import copy
import importlib
import subprocess
import pandas as pd

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
