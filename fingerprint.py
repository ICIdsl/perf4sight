import os
import sys
import glob
import time
import pickle
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import src.adapt.performance_prediction.performance_model.data_fitter as linear_fitter
import src.adapt.performance_prediction.performance_model.decision_tree_fitter as dt_fitter
import src.adapt.performance_prediction.performance_model.pass_fail_decision_tree as pf_fitter
import src.adapt.performance_prediction.performance_model.classifier_decision_tree as pf_fitter_1

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

def get_data_pts(log, targetVar, fitter, filterPassed=lambda x: x.passed):
#{{{
    data = []
    for row in log.iterrows():
        if filterPassed(row[1]):
            batchSize = row[1].bs
            modelFileName = row[1].model.replace('ar4414/memory_model', 'adapt/performance_prediction')
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

def backfill_failed_memory_consumption(params, nets, pruningLevels):
#{{{
    def backfill(log, var):
        decTreeFile = os.path.join(params.fingerprint['loc'], f"{modelVersion}_{var}.pkl")
        with open(decTreeFile, 'rb') as f:
            decTree = pickle.load(f) 
        fitter = dt_fitter.Fitter(modelVersion)
        modelFileName = lambda row : row.model.replace('ar4414/memory_model', 'adapt/performance_prediction')
        fitVars = [fitter.get_fit_variables(row.bs, row.net_name, modelFileName(row), row.dataset)\
                for idx,row in log.iterrows()]
        predictions = decTree.predict(fitVars)
        log.loc[:,var] = predictions
        return log

    modelVersion = params.fingerprint['performance_model_version'].split('_')[0]
    for net in nets: 
        logPath = 'memory_logs'
        log = get_logs(net, params.fingerprint[logPath], pruningLevels)
            
        for model,data in log.groupby('model'): 
            testName = params.fingerprint['loc'].split('/')[-1]
            fileName = model.split('/')[-1].split('.')[0]
            fileName = f"{fileName}-backfilled-{testName}.csv"
            filePath = os.path.join(params.fingerprint['memory_logs'], fileName)
            
            backfillData = False
            if os.path.isfile(filePath): 
                if eval(params.fingerprint['overwrite']): 
                    backfillData = True
            else:
                backfillData = True
            
            if backfillData:
                data = pd.DataFrame(data)
                print(f"Backfilling data for {fileName}")
                for var in ['mem_alloc', 'mem_reserved']:
                    data = backfill(data, var)
                
                if eval(params.fingerprint['backfill_mem_avail']):
                    memAvail = data.loc[data.passed == True].mem_avail.mean()
                    failIdx = data.loc[data.passed == False].index
                    data.loc[failIdx, 'mem_avail'] = memAvail
                
                print(f"Writing backfilled log to {filePath}")
                data.to_csv(filePath, index=False, sep='|')
#}}}

def create_decision_tree_model(params, var, memory=True): 
#{{{
    modelVersion = params.fingerprint['performance_model_version'].split('_')[0]
    dirName = params.fingerprint['loc']
    fileName = "{}_{}.pkl".format(modelVersion, var)
    saveFile = os.path.join(dirName, fileName)
    
    createModel = False
    if os.path.isfile(saveFile):
        if eval(params.fingerprint['overwrite']):
            createModel = True 
    else:
        createModel = True
    
    if createModel:
        print(f"Creating decision tree model for {fileName}")
        fitter = dt_fitter.Fitter(modelVersion)
        trainData = []
        for net in eval(params.fingerprint['train_nets']): 
            logType = 'memory_logs' if memory else 'latency_logs'
            for logPath in eval(params.fingerprint[logType]):
                # log = get_logs(net, params.fingerprint[logPath], eval(params.fingerprint['train_logs']))
                log = get_logs(net, logPath, eval(params.fingerprint['train_logs']))
                trainData += get_data_pts(log, var, fitter)
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
    modelVersion = params.fingerprint['performance_model_version'].split('_')[0]
    dirName = params.fingerprint['loc']
    fileName = "{}_{}.pkl".format(modelVersion, var)
    filePath = os.path.join(dirName, fileName)
    with open(filePath, 'rb') as f: 
        decTree = pickle.load(f)

    evalData = []
    fitter = dt_fitter.Fitter(modelVersion)
    for net in eval(params.fingerprint['eval_nets']): 
        if memory:
            logPath = 'eval_memory_logs' if 'eval_memory_logs' in params.fingerprint.keys() else 'memory_logs'
        else:
            logPath = 'eval_latency_logs' if 'eval_latency_logs' in params.fingerprint.keys() else 'latency_logs'
        log = get_logs(net, params.fingerprint[logPath], eval(params.fingerprint['eval_logs']))
        evalData += get_data_pts(log, var, fitter)
        
    modelType = 'memory' if memory else 'latency'
    print(f"Evaluating {modelType} model for variable {var}")
    fitter.evaluate(decTree, evalData) 
#}}}

def create_fit_classification_model(params, backfill=True): 
#{{{
    dirName = params.fingerprint['loc']
    fileName = f"fit_classifier_model_based_{params.fingerprint['classifier_model_version']}.pkl"
    saveFile = os.path.join(dirName, fileName)

    fitClassifier = False
    if os.path.isfile(saveFile): 
        if eval(params.fingerprint['overwrite']): 
            fitClassifier = True
    else:
        fitClassifier = True
    
    if fitClassifier:
        print(f"Creating fit classifier")
        trainData = []
        for net in eval(params.fingerprint['train_nets']): 
            testName = params.fingerprint['loc'].split('/')[-1]
            if backfill:
                extention = f"backfilled-{testName}"
                log = get_logs(net, params.fingerprint['memory_logs'], eval(params.fingerprint['train_logs']), extention)
            else:
                log = get_logs(net, params.fingerprint['memory_logs'], eval(params.fingerprint['train_logs']))
            trainData.append(log)
        trainData = pd.concat(trainData, ignore_index=True)
        
        fitter = pf_fitter.Fitter(params.fingerprint['classifier_model_version'])
        features, featNames = fitter.get_fit_variables(trainData)
        decTree = fitter.decision_tree(features, featNames)
        
        cmd = f"mkdir -p {dirName}"
        subprocess.check_call(cmd, shell=True)
        print("Writing model to {}".format(saveFile))
        with open(saveFile, 'wb') as pklFile:
            pickle.dump(decTree, pklFile)
    
    return fitClassifier
#}}}

def test_classification_model(params):
#{{{
    dirName = params.fingerprint['loc']
    fileName = f"fit_classifier_model_based_{params.fingerprint['classifier_model_version']}.pkl"
    modelFile = os.path.join(dirName, fileName)
    with open(modelFile, 'rb') as f:
        decTree = pickle.load(f)

    evalData = []
    for net in eval(params.fingerprint['eval_nets']): 
        testName = params.fingerprint['loc'].split('/')[-1]
        extension = f"backfilled-{testName}"
        log = get_logs(net, params.fingerprint['memory_logs'], eval(params.fingerprint['eval_logs']), extension)
        evalData.append(log)
    evalData = pd.concat(evalData, ignore_index=True, sort=True)
    
    print(f"Evaluating memory fit classification model")
    fitter = pf_fitter.Fitter(params.fingerprint['classifier_model_version'])
    features, featNames = fitter.get_fit_variables(evalData)
    fitter.evaluate(decTree, features)
#}}}

def fingerprint_device(params):
#{{{
    perfModelType = params.fingerprint['performance_model_type']
    cModelType = params.fingerprint['classifier_model_type']
    memTree = []
    latTree = []
    newClassifier = False
    if not eval(params.fingerprint['evaluate']):
    #{{{
        # build memory models
        for var in eval(params.fingerprint['memory_targets']): 
            if  perfModelType == 'dec_tree':
                memTree.append(create_decision_tree_model(params, var))
            elif perfModelType == 'none':
                print(f"Not creating memory models")

        # build latency model
        for var in eval(params.fingerprint['latency_targets']): 
            if perfModelType == 'dec_tree':
                latTree.append(create_decision_tree_model(params, var, False))
            elif perfModelType == 'none':
                print(f"Not creating latency model")
   
        # build classifiers
        if cModelType == 'perf_model_based':
            nets = eval(params.fingerprint['train_nets'])
            pl = eval(params.fingerprint['train_logs'])
            backfill_failed_memory_consumption(params, nets, pl)
            newClassifier = create_fit_classification_model(params)
        elif cModelType == 'no_backfill':
            newClassifier = create_fit_classification_model(params)
        elif cModelType == 'none':
            newClassifier = False
            print("No classification model fit")
        else:
            print(f"Classification model type - {cModelType} unknown")
            newClassifier = False
    #}}}
    
    # test memory models
    if any(memTree) or eval(params.fingerprint['evaluate']):
    #{{{
        if perfModelType == 'dec_tree': 
            [test_regression_model(params, var) for var in eval(params.fingerprint['memory_targets'])]
        elif perfModelType == 'none':
            print(f"Not evaluating memory models")
        else:
            print(f"Testing not implemented for performance model type {perfModelType}")
    #}}}
    
    # test latency models
    if any(latTree) or eval(params.fingerprint['evaluate']):
    #{{{
        if perfModelType == 'dec_tree':
            [test_regression_model(params, var, False) for var in eval(params.fingerprint['latency_targets'])]
        elif perfModelType == 'none':
            print(f"Not evaluating latency models")
        else:
            print(f"Testing not implemented for performance model type {perfModelType}")
    #}}}

    # test classifier
    if newClassifier or eval(params.fingerprint['evaluate']):
    #{{{
        if cModelType == 'perf_model_based' or cModelType == 'no_backfill':
            nets = eval(params.fingerprint['eval_nets'])
            pl = eval(params.fingerprint['eval_logs'])
            backfill_failed_memory_consumption(params, nets, pl)
            test_classification_model(params)
        elif cModelType == 'none':
            print("Classification model not fit")
        else:
            print(f"Cannot evaluate unknown classification model type - {cModelType}")
    #}}}
#}}}
