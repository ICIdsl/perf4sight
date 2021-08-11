import os
import sys
import glob
import time
import pickle
import subprocess

import numpy as np
import pandas as pd

import fingerprinting.decision_tree_fitter as dt_fitter

def get_logs(net, logPath, pruningLevels, ext="gt"): 
#{{{
    logs = []
    for pp in pruningLevels:
        netWildcard = f"{net}_*_{pp}-{ext}.csv" if 'ofa' in net else "{}_{}-{}.csv".format(net, pp, ext) 
        path = os.path.join(logPath, netWildcard)
        try:
            for f in glob.glob(path):
                logs.append(pd.read_csv(f, delimiter='|'))
        except:
            print("WARNING: Could not find log for {}. Check file path or profile data".format(path))
    log = pd.concat(logs, ignore_index=True).sort_values(by='bs')
    return log
#}}}

def get_data_pts(log, prunedModelDir, targetVar, fitter, filterPassed=lambda x: x.passed):
#{{{
    data = []
    for row in log.iterrows():
        if filterPassed(row[1]):
            batchSize = row[1].bs
            modelFile = row[1].model.split('/')[-1]
            modelFileName = os.path.join(prunedModelDir, modelFile)
            if 'ofa' in row[1].model:
                fName = modelFileName.split('/')[-1]
                samp = fName.split('_')[2]
                netName = f'ofaresnet50_{samp}'
            else:
                netName = row[1].net_name
            var = fitter.get_fit_variables(batchSize, netName, modelFileName, row[1].dataset)
            data.append((*var, row[1][targetVar]))
    return data
#}}}

def create_decision_tree_model(params, var, memory=True): 
#{{{
    modelVersion = params.fingerprint['stage']
    dirName = params.fingerprint['model_storedir']
    fileName = "{}_stage_{}.pkl".format(modelVersion, var)
    prunedModelStoredir = params.fingerprint['pruned_models_storedir']
    
    createModel = True 
    saveFile = os.path.join(dirName, fileName)
    if os.path.isfile(saveFile) and not eval(params.fingerprint['overwrite']):
        print(f"Model already exists at {saveFile}")
        return True
    
    if createModel:
        print(f"Creating decision tree model for {fileName}")
        trainData = []
        fitter = dt_fitter.Fitter(modelVersion)
        for net in eval(params.fingerprint['train_nets']): 
            logType = 'memory_logs' if memory else 'latency_logs'
            for logPath in eval(params.fingerprint[logType]):
                log = get_logs(net, logPath, eval(params.fingerprint['train_logs']))
                trainData += get_data_pts(log, prunedModelStoredir, var, fitter)
        trainPred, decTree = fitter.random_forest(trainData)
        
        cmd = f"mkdir -p {dirName}"
        subprocess.check_call(cmd, shell=True)
        print("Writing model to {}".format(saveFile))
        with open(saveFile, 'wb') as pklFile:
            pickle.dump(decTree, pklFile)

    return createModel
#}}}

def test_regression_model(params, var, memory=True):
#{{{
    modelVersion = params.fingerprint['stage']
    dirName = params.fingerprint['model_storedir']
    fileName = "{}_stage_{}.pkl".format(modelVersion, var)
    prunedModelStoredir = params.fingerprint['pruned_models_storedir']
    
    filePath = os.path.join(dirName, fileName)
    with open(filePath, 'rb') as f: 
        decTree = pickle.load(f)

    evalData = []
    fitter = dt_fitter.Fitter(modelVersion)
    modelType = 'memory' if memory else 'latency'
    for net in eval(params.fingerprint['eval_nets']): 
        if len(eval(params.fingerprint[f'eval_{modelType}_logs'])) != 0:
            logPaths = eval(params.fingerprint[f'eval_{modelType}_logs'])
        else:
            logPaths = eval(params.fingerprint[f'{modelType}_logs'])
        
        for logPath in logPaths:
            log = get_logs(net, logPath, eval(params.fingerprint['eval_logs']))
            evalData += get_data_pts(log, prunedModelStoredir, var, fitter)
        
    print(f"Evaluating {modelType} model for variable {var}")
    fitter.evaluate(decTree, evalData) 
#}}}

def fingerprint_device(params):
#{{{
    memModel = False
    latModel = False 
    if not eval(params.fingerprint['evaluate']):
    #{{{
        # build memory models
        if eval(params.fingerprint['memory_model']):
            memModel = create_decision_tree_model(params, 'mem_used', memory=True)
        else:
            print(f"Not creating memory model")

        # build latency model
        if eval(params.fingerprint['latency_model']):
            latModel = create_decision_tree_model(params, 'latency', memory=False)
        else:
            print(f"Not creating latency model")
    #}}}
    
    # test memory models
    if memModel or eval(params.fingerprint['evaluate']):
        test_regression_model(params, 'mem_used', memory=True)
    
    # test latency models
    if latModel or eval(params.fingerprint['evaluate']):
        test_regression_model(params, 'latency', memory=False)
#}}}
