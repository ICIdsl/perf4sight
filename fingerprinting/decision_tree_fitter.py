import os
import sys
import time 
import math
import configparser as cp

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import utils 

def training(config):
#{{{
    varList = []
    featNames = []
    
    op = 1 + math.floor((config['ip'] + 2*config['pad'] - config['k']) / config['stride']) 
    config['op'] = op
    effIc = (int(config['ic'])/int(config['groups']))
    config['effIc'] = effIc

    ###### Tensors allocations 
    #{{{
    # weights 
    weights = int(config['oc'] * effIc * config['k']**2)
    varList.append(weights)
    featNames.append('tensor-weights')
    
    # weight gradient
    weightGrad = int(config['bs'] * config['oc'] * effIc * config['k']**2)
    varList.append(weightGrad)
    featNames.append('tensor-weight_grad')
    
    # input gradient + ifm
    ifm = int(2 * config['bs'] * config['ic'] * config['ip']**2)
    varList.append(ifm)
    featNames.append('tensor-ifm')

    # output gradient + ofm
    ofm = int(2 * config['bs'] * config['oc'] * op**2)
    varList.append(ofm)
    featNames.append('tensor-ofm')

    # combination of tensor allocations
    sum1 = weights + weightGrad + ifm + ofm
    varList.append(sum1)
    featNames.append('tensors-sum')
    #}}}
    
    ###### GEMM implementation 
    #{{{
    # i2c-fwd-indices
    i2cIdxFwd = int(config['bs'] * op**2)
    varList.append(i2cIdxFwd)
    featNames.append('gemm-i2c-idx-fwd')
    
    # i2c-fwd-total
    i2cFwd = int(config['bs'] * op**2 * config['k']**2 * config['ic'])
    varList.append(i2cFwd)
    featNames.append('gemm-i2c-fwd')
    
    # i2c-bwd_x-indices
    i2cIdxBwdX = int(config['bs'] * config['ip']**2)
    varList.append(i2cIdxBwdX)
    featNames.append('gemm-i2c-idx-bwd-x')
    
    # i2c-bwd_x-total
    i2cBwdX = int(config['bs'] * config['ip']**2 * config['k']**2 * config['ic'])
    varList.append(i2cBwdX)
    featNames.append('gemm-i2c-bwd-x')
    
    # i2c-bwd_w-total
    i2cBwdW = int(config['bs'] * op**2 * config['k']**2 * effIc)
    varList.append(i2cBwdW)
    featNames.append('gemm-i2c-bwd-w')

    varList.append(i2cFwd + i2cBwdX  +i2cBwdW)
    featNames.append('gemm-i2c-sum-total')
    
    varList.append(2*i2cIdxFwd + i2cIdxBwdX)
    featNames.append('gemm-i2c-index-sum-total')
    #}}}
    
    ###### GEMM ops 
    #{{{
    gemmOpsFwd = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
    varList.append(gemmOpsFwd)
    featNames.append('gemm-ops-fwd')
    
    # gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['op'] 
    gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['oc'] 
    varList.append(gemmOpsBwdX)
    featNames.append('gemm-ops-bwd')

    gemmOps = 2*gemmOpsFwd + gemmOpsBwdX
    varList.append(gemmOps)
    featNames.append('gemm-ops-sum')
    #}}}

    ###### FFT implementation 
    #{{{
    # fft weights fwd
    fftWeightsFwd = int(config['ip'] * (1+config['ip']) * config['oc'] * effIc)
    varList.append(fftWeightsFwd)
    featNames.append('fft-weights-fwd')
    
    # fft ifm fwd
    fftIfmFwd = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
    varList.append(fftIfmFwd)
    featNames.append('fft-ifm-fwd')
    
    # fft ofm bwdw
    fftOfmBwdW = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['oc'])) 
    varList.append(fftOfmBwdW)
    featNames.append('fft-ofm-bwd-w')
    
    # fft weights bwdx
    fftWeightsBwdX = int(op * (1+op) * config['oc'] * effIc)
    varList.append(fftWeightsBwdX)
    featNames.append('fft-weights-bwd-x')

    # fft ofm bwdx
    fftOfmBwdX = int(op * (1+op) * (config['bs'] * config['oc'])) 
    varList.append(fftOfmBwdX)
    featNames.append('fft-ofm-bwd-x')
    
    # fft-sum-1
    varList.append(fftIfmFwd + fftWeightsFwd)
    featNames.append('fft-sum-fwd')
    
    # fft-sum-2
    varList.append(fftOfmBwdX + fftWeightsBwdX)
    featNames.append('fft-sum-bwd_x')
    
    # fft-sum-3
    fftIfmBwdW = fftIfmFwd
    varList.append(fftOfmBwdW + fftIfmBwdW)
    featNames.append('fft-sum-bwd_w')

    # fft-sum-all
    varList.append(fftIfmFwd + fftWeightsFwd + fftOfmBwdX + fftWeightsBwdX + fftOfmBwdW + fftIfmBwdW)
    featNames.append('fft-sum-all')
    #}}}
    
    ###### FFT ops
    #{{{
    fftFwdOps = (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
            config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
    varList.append(fftFwdOps)
    featNames.append('fft-fwd-ops')
    
    fftBwdXOps = (op**2 * math.log(op,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
            config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * op**2)
    varList.append(fftBwdXOps)
    featNames.append('fft-bwd_x-ops')
    
    fftBwdWOps = (config['ip'] * math.log(config['ip']**2,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
            config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
    varList.append(fftBwdWOps)
    featNames.append('fft-bwd_w-ops')

    varList.append(fftFwdOps + fftBwdXOps + fftBwdWOps)
    featNames.append('fft-ops-sum')
    #}}}
    
    ###### Winograd implementation 
    #{{{
    def wino_mem(m,r):
    #{{{
        # winograd forward
        fwd = int(3 * config['bs'] * config['oc'] * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
        # winograd tiles grad wrt inputs 
        bwdX = int(3 * config['bs'] * config['ic'] * (m+r-1)**2 * math.ceil(op/m)**2)
        # winograd tiles grad wrt weights
        bwdW = int(3 * config['bs'] * config['oc'] * effIc * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
        
        varList.append(fwd)
        featNames.append(f'wino-fwd-{m}-{r}')
        varList.append(bwdX)
        featNames.append(f'wino-bwd_x-{m}-{r}')
        varList.append(bwdW)
        featNames.append(f'wino-bwd_w-{m}-{r}')
        
        # winograd sum-1
        varList.append(fwd + bwdX)
        featNames.append(f'wino-sum-1-{m}-{r}')
        # winograd sum-2
        varList.append(fwd + bwdW)
        featNames.append('wino-sum-2-{m}-{r}')
        # winograd sum-3
        varList.append(bwdX + bwdW)
        featNames.append('wino-sum-3-{m}-{r}')
        # winograd sum-all
        varList.append(fwd + bwdX + bwdW)
        featNames.append('wino-sum-all-{m}-{r}')
    #}}}
    
    wino_mem(4,3)
    wino_mem(3,2)
    #}}}

    ###### Winograd ops
    #{{{
    def wino_ops(m,r): 
    #{{{
        fwd = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
        bwdX = config['bs'] * config['ic'] * config['oc'] * math.ceil(op/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
        bwdW = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['op']/r)**2 * (m+r-1)**2
        
        featNames.append(f'wino-ops-fwd-{m}-{r}')
        varList.append(fwd)
        featNames.append(f'wino-ops-bwd_x-{m}-{r}')
        varList.append(bwdX)
        featNames.append(f'wino-ops-bwd_w-{m}-{r}')
        varList.append(bwdW)
        featNames.append(f'wino-ops-sum-1-{m}-{r}')
        varList.append(fwd + bwdX)
        featNames.append(f'wino-ops-sum-2-{m}-{r}')
        varList.append(fwd + bwdW)
        featNames.append(f'wino-ops-sum-3-{m}-{r}')
        varList.append(bwdX + bwdW)
        featNames.append(f'wino-ops-sum-all-{m}-{r}')
        varList.append(fwd + bwdX + bwdW)
    #}}}
    
    wino_ops(4,3)
    wino_ops(3,2)
    #}}}
    
    return featNames, varList
#}}}
    
def inference(config):
#{{{
    varList = []
    featNames = []
    
    op = 1 + math.floor((config['ip'] + 2*config['pad'] - config['k']) / config['stride']) 
    config['op'] = op
    effIc = (int(config['ic'])/int(config['groups']))
    config['effIc'] = effIc

    ###### Tensors allocations 
    #{{{
    # weights 
    weights = int(config['oc'] * effIc * config['k']**2)
    varList.append(weights)
    featNames.append('tensor-weights')
    
    # ifm
    ifm = int(config['bs'] * config['ic'] * config['ip']**2)
    varList.append(ifm)
    featNames.append('tensor-ifm')

    # ofm
    ofm = int(config['bs'] * config['oc'] * op**2)
    varList.append(ofm)
    featNames.append('tensor-ofm')

    # combination of tensor allocations
    sum1 = weights + ifm + ofm
    varList.append(sum1)
    featNames.append('tensors-sum')
    #}}}
    
    ###### GEMM implementation 
    #{{{
    # i2c-fwd-indices
    i2cIdxFwd = int(config['bs'] * op**2)
    varList.append(i2cIdxFwd)
    featNames.append('gemm-i2c-idx-fwd')
    
    # i2c-fwd-total
    i2cFwd = int(config['bs'] * op**2 * config['k']**2 * config['ic'])
    varList.append(i2cFwd)
    featNames.append('gemm-i2c-fwd')
    #}}}
    
    ###### GEMM ops 
    #{{{
    gemmOpsFwd = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
    varList.append(gemmOpsFwd)
    featNames.append('gemm-ops-fwd')
    #}}}

    ###### FFT implementation 
    #{{{
    # fft weights fwd
    fftWeightsFwd = int(config['ip'] * (1+config['ip']) * config['oc'] * effIc)
    varList.append(fftWeightsFwd)
    featNames.append('fft-weights-fwd')
    
    # fft ifm fwd
    fftIfmFwd = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
    varList.append(fftIfmFwd)
    featNames.append('fft-ifm-fwd')
    
    # fft-sum-1
    varList.append(fftIfmFwd + fftWeightsFwd)
    featNames.append('fft-sum-fwd')
    #}}}
    
    ###### FFT ops
    #{{{
    fftFwdOps = (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
            config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
    varList.append(fftFwdOps)
    featNames.append('fft-fwd-ops')
    #}}}
    
    ###### Winograd implementation 
    #{{{
    def wino_mem(m,r):
    #{{{
        # winograd forward
        fwd = int(3 * config['bs'] * config['oc'] * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
        varList.append(fwd)
        featNames.append(f'wino-fwd-{m}-{r}')
    #}}}
    
    wino_mem(4,3)
    wino_mem(3,2)
    #}}}

    ###### Winograd ops
    #{{{
    def wino_ops(m,r): 
    #{{{
        fwd = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
        featNames.append(f'wino-ops-fwd-{m}-{r}')
        varList.append(fwd)
    #}}}
    
    wino_ops(4,3)
    wino_ops(3,2)
    #}}}
    
    return featNames, varList
#}}}
    
class Fitter(): 
    def __init__(self, modelVersion): 
        self.bs = []
        self.modelFunc = training if modelVersion == 'training' else inference

    def get_fit_variables(self, batchSize, netName, modelFileName, dataset): 
    #{{{
        # file uniquified using last modification time of generated pruned model
        # this way if the model is pruned again, the description file will be changed
        netFile = "{}_{}.ini".format(modelFileName.split('/')[-1].split('.')[0], int(os.path.getmtime(modelFileName)))
        netDescFile = "{}/{}/{}".format(os.path.dirname(os.path.realpath(__file__)), 'layer_desc_imagenet', netFile)
        if not os.path.isfile(netDescFile):
            print(f"Creating layer description {netDescFile}")
            model = utils.read_model(modelFileName, netName) 
            testGenerator = utils.TestGenerator()
            testGenerator.from_model(model, dataset, netDescFile)  
       
        modelDesc = cp.ConfigParser()
        modelDesc.read(netDescFile)
        nonAccVars = {}
        for i, layer in enumerate(modelDesc.sections()): 
            if 'mode' in modelDesc[layer].keys():
                modelDesc[layer].pop('mode') 
            config = {k:eval(v) for k,v in dict(modelDesc[layer]).items()}
            config = {k:v[0] if type(v) is list or type(v) is tuple else v for k,v in config.items()}
            config['bs'] = batchSize
            
            self.featureNames, memPreds = self.modelFunc(config)
            if i == 0:
                predictions = np.array(memPreds)
            else:
                predictions += memPreds
        
        predictions = list(predictions)
        
        self.bs.append(batchSize)
        return predictions
    #}}}

    def random_forest(self, data): 
    #{{{
        print("Num Training Samples = {}".format(len(data)))
        # normalise 
        cases = [x[0:-1] for x in data]
        gt = [float(x[-1]) for x in data]
        start = time.time()
        decTree = ExtraTreesRegressor(n_estimators=300, n_jobs=-1)
        decTree = decTree.fit(cases, gt)
        print(f"Training took {time.time() - start}s")
        print("Score = {:.2f}".format(decTree.score(cases, gt)))
        
        predictions = decTree.predict(cases)
        trainErr = [abs(predictions[i] - gt[i])/gt[i] for i in range(len(gt))]
        print("Mean training error = {:.2f}".format(np.mean(trainErr)))

        return predictions, decTree
    #}}}

    def evaluate(self, decTree, data): 
    #{{{
        print("Num Testing Samples = {}".format(len(data)))
        
        cases = [x[0:-1] for x in data]
        gt = [float(x[-1]) for x in data]
        r2 = decTree.score(cases, gt)
        print("Evaluation score = {:.2f}".format(r2))
        
        start = time.time()
        predictions = decTree.predict(cases)
        
        print(f"Prediction took {time.time() - start}s")
        testErr = [abs(predictions[i] - gt[i])/gt[i] for i in range(len(gt))]
        self.meanErr = np.mean(testErr)
        print("Mean test error = {:.4f}".format(self.meanErr))
        
        return predictions
    #}}}







