import graphviz
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import os
import sys
import time 
import math
import configparser as cp
from abc import ABC, abstractmethod

currDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currDir)

import utils as utils
from test_generator import TestGenerator

class BasicConv(): 
    @classmethod
    def model_v0(cls, config):
    #{{{
        op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
        
        # weights 
        var1 = int(config['oc'] * (int(config['ic'])/int(config['groups'])) * config['k']**2)
        # i2c-indices
        var2 = int(config['bs'] * op**2)
        # input gradient + ifm
        var3 = int(config['bs'] * config['ic'] * config['ip']**2)
        # weight gradient
        var4 = int(config['bs'] * config['oc'] * (int(config['ic'])/int(config['groups'])) * config['k']**2)
        # output gradient + ofm
        var5 = int(config['bs'] * config['oc'] * op**2)

        # fft ifm 
        var6 = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
        # fft ofm
        var7 = int(op * (1+op) * (config['bs'] * config['oc'])) 
        # fft weights
        var8 = int(config['ip'] * (1+config['ip']) * (config['oc'] * (int(config['ic'])/int(config['groups']))))

        gIp = int(config['ic'])/int(config['groups'])
        # winograd forward
        m = op 
        r = config['k'] 
        var9 = int(gIp * (m+r-1)**2)
        
        # winograd tiles grad wrt inputs 
        m = config['ip']
        r = config['k']
        var10 = int(config['oc'] * (m+r-1)**2)

        # winograd tiles grad wrt weights
        # m = config['k']
        # r = config['ip']
        # var11 = int(config['oc'] * (m+r-1)**2)

        featureNames = ['weights', 'i2c-idx', 'ifm', 'weight-grad', 'ofm', 'fft-ifm', 'fft-ofm', 'fft-weights', 'wino-fwd', 'wino-ipgrad']
        
        return featureNames, [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10]
    #}}}
    
    @classmethod
    def model_v1(cls, config):
    #{{{
        varList = []
        featNames = []
        
        op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
        config['op'] = op
        effIc = (int(config['ic'])/int(config['groups']))
        config['effIc'] = effIc

        ###### single vars
        for var in ['bs', 'ic', 'effIc', 'ip', 'oc', 'op', 'k', 'groups', 'stride', 'pad']:
            varList.append(config[var])
            featNames.append(var)

        ###### tensors allocations 
        # weights 
        weights = int(config['oc'] * (int(config['ic'])/int(config['groups'])) * config['k']**2)
        varList.append(weights)
        featNames.append('tensor-weights')
        # weight gradient
        weightGrad = int(config['bs'] * config['oc'] * (int(config['ic'])/int(config['groups'])) * config['k']**2)
        varList.append(weightGrad)
        featNames.append('tensor-weight_grad')
        
        # input gradient + ifm
        ifm = int(config['bs'] * config['ic'] * config['ip']**2)
        varList.append(ifm)
        featNames.append('tensor-ifm')

        # output gradient + ofm
        ofm = int(config['bs'] * config['oc'] * op**2)
        varList.append(ofm)
        featNames.append('tensor-ofm')

        # combination of tensor allocations
        sum1 = weights + weightGrad + 2*ifm + 2*ofm
        varList.append(sum1)
        featNames.append('tensors-sum')

        ###### GEMM implementation 
        # i2c-indices
        i2cIdx = int(config['bs'] * op**2)
        varList.append(i2cIdx)
        featNames.append('gemm-i2c-idx')

        ###### FFT implementation 
        # fft ifm 
        fftIfm = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
        varList.append(fftIfm)
        featNames.append('fft-ifm')
        
        # fft ofm
        fftOfm = int(op * (1+op) * (config['bs'] * config['oc'])) 
        varList.append(fftOfm)
        featNames.append('fft-ofm')
        
        # fft weights
        fftWeights = int(config['ip'] * (1+config['ip']) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeights)
        featNames.append('fft-weights')

        # fft-sum-1
        varList.append(fftIfm + fftOfm)
        featNames.append('fft-sum-ifm-ofm')
        
        # fft-sum-2
        varList.append(fftIfm + fftWeights)
        featNames.append('fft-sum-ifm-weights')
        
        # fft-sum-3
        varList.append(fftOfm + fftWeights)
        featNames.append('fft-sum-ofm-weights')

        # fft-sum-all
        varList.append(fftIfm + fftOfm + fftWeights)
        featNames.append('fft-sum-all')
        
        ###### Winograd implementation 
        # winograd forward
        m = op 
        r = config['k'] 
        winoFwd = int(effIc * (m+r-1)**2)
        varList.append(winoFwd)
        featNames.append('wino-fwd')
        
        # winograd tiles grad wrt inputs 
        m = config['ip']
        r = config['k']
        winoIfmGrad = int(config['oc'] * (m+r-1)**2)
        varList.append(winoIfmGrad)
        featNames.append('wino-gradIfm')

        # winograd sum
        varList.append(winoFwd + winoIfmGrad)
        featNames.append('wino-sum')

        # winograd tiles grad wrt weights
        # m = config['k']
        # r = config['ip']
        # var11 = int(config['oc'] * (m+r-1)**2)
        
        return featNames, varList
    #}}}
    
    @classmethod
    def model_v2(cls, config):
    #{{{
        varList = []
        featNames = []
        
        op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
        config['op'] = op
        effIc = (int(config['ic'])/int(config['groups']))
        config['effIc'] = effIc

        ###### single vars
        for var in ['bs', 'ic', 'effIc', 'ip_2', 'oc', 'op_2', 'k_2', 'groups', 'stride', 'pad']:
            if '2' in var:
                var1 = var.split('_')[0]
                varList.append(config[var1]**2)
            else:
                varList.append(config[var])
            featNames.append(var)
        # for var in ['bs', 'ic', 'effIc', 'ip', 'oc', 'op', 'k', 'groups', 'stride', 'pad']:
        #     varList.append(config[var])
        #     featNames.append(var)

        ###### tensors allocations 
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
        
        ###### GEMM implementation 
        # i2c-indices
        i2cIdx = int(config['bs'] * op**2)
        varList.append(i2cIdx)
        featNames.append('gemm-i2c-idx')
        
        # i2c-total
        i2c = int(config['bs'] * op**2 * config['k']**2 * config['ic'])
        varList.append(i2c)
        featNames.append('gemm-i2c')

        ###### FFT implementation 
        # fft ifm 
        fftIfm = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
        varList.append(fftIfm)
        featNames.append('fft-ifm')
        
        # fft ifm 
        fftIfmBwd = int(op * (1+op) * (config['bs'] * config['ic'])) 
        varList.append(fftIfmBwd)
        featNames.append('fft-ifm-bwd')
        
        # fft ofm
        fftOfm = int(op * (1+op) * (config['bs'] * config['oc'])) 
        varList.append(fftOfm)
        featNames.append('fft-ofm')
        
        # fft weights
        fftWeights = int(config['ip'] * (1+config['ip']) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeights)
        featNames.append('fft-weights')
        
        # fft weights
        fftWeightsBwd = int(op * (1+op) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeightsBwd)
        featNames.append('fft-weights-bwd')

        # fft-sum-1
        varList.append(fftIfmBwd + fftOfm)
        featNames.append('fft-sum-ifm-bwd-ofm')
        
        # fft-sum-2
        varList.append(fftIfm + fftWeights)
        featNames.append('fft-sum-ifm-weights')
        
        # fft-sum-3
        varList.append(fftOfm + fftWeightsBwd)
        featNames.append('fft-sum-ofm-weights-bwd')

        # fft-sum-all
        varList.append(fftIfm + fftIfmBwd + fftOfm + fftWeights + fftWeightsBwd)
        featNames.append('fft-sum-all')
        
        ###### Winograd implementation 
        # winograd forward
        m = op 
        r = config['k'] 
        winoFwd = int(effIc * (m+r-1)**2)
        varList.append(winoFwd)
        featNames.append('wino-fwd')
        
        # winograd tiles grad wrt inputs 
        m = config['ip']
        r = config['k']
        winoIfmGrad = int(config['oc'] * (m+r-1)**2)
        varList.append(winoIfmGrad)
        featNames.append('wino-gradIfm')

        # winograd tiles grad wrt weights
        m = config['k'] 
        r = config['ip']
        winoWeightGrad = int((m+r-1)**2)
        varList.append(winoWeightGrad)
        featNames.append('wino-gradWeight')

        # winograd sum-1
        varList.append(winoFwd + winoIfmGrad)
        featNames.append('wino-sum-1')
        
        # winograd sum-2
        varList.append(winoFwd + winoWeightGrad)
        featNames.append('wino-sum-2')
        
        # winograd sum-3
        varList.append(winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-3')
        
        # winograd sum-all
        varList.append(winoFwd + winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-all')
        
        return featNames, varList
    #}}}
    
    @classmethod
    def model_v3(cls, config, nonAccVars):
    #{{{
        varList = []
        featNames = []
        
        op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
        config['op'] = op
        effIc = (int(config['ic'])/int(config['groups']))
        config['effIc'] = effIc

        ###### single vars
        for var in ['bs', 'ic', 'effIc', 'ip', 'oc', 'op', 'k', 'groups', 'stride', 'pad']:
            if var in nonAccVars.keys():
                nonAccVars[var].append(config[var])
            else:
                nonAccVars[var] = [config[var]]
        
        ###### GEMM ops 
        gemmOps = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
        varList.append(gemmOps)
        featNames.append('gemm-ops')

        ###### fft ops
        fftFwd = 2 * (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        varList.append(fftFwd)
        featNames.append('fft-fwd-ops')
        
        fftBwdIfm = (op**2 * math.log(op,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * op**2)
        varList.append(fftBwdIfm)
        featNames.append('fft-bwd-ifm-ops')

        ###### winograd ops
        m = op 
        r = config['k'] 
        opsFwd = effIc * (m+r-1) * r**2
        varList.append(opsFwd)
        featNames.append('wino-fwd-ops')
        
        m = config['ip']
        r = config['k']
        opsBwdIfm = config['oc'] * (m+r-1) * r**2
        varList.append(opsBwdIfm)
        featNames.append('wino-bwd-ifm-ops')
        
        m = config['k'] 
        r = config['ip']
        opsBwdWeights = (m+r-1) * r**2
        varList.append(opsBwdWeights)
        featNames.append('wino-bwd-weights-ops')

        ###### tensors allocations 
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
        
        ###### GEMM implementation 
        # i2c-indices
        i2cIdx = int(config['bs'] * op**2)
        varList.append(i2cIdx)
        featNames.append('gemm-i2c-idx')
        
        # i2c-total
        i2c = int(config['bs'] * op**2 * config['k']**2 * config['ic'])
        varList.append(i2c)
        featNames.append('gemm-i2c')

        ###### FFT implementation 
        # fft ifm 
        fftIfm = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
        varList.append(fftIfm)
        featNames.append('fft-ifm')
        
        # fft ifm 
        fftIfmBwd = int(op * (1+op) * (config['bs'] * config['ic'])) 
        varList.append(fftIfmBwd)
        featNames.append('fft-ifm-bwd')
        
        # fft ofm
        fftOfm = int(op * (1+op) * (config['bs'] * config['oc'])) 
        varList.append(fftOfm)
        featNames.append('fft-ofm')
        
        # fft weights
        fftWeights = int(config['ip'] * (1+config['ip']) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeights)
        featNames.append('fft-weights')
        
        # fft weights
        fftWeightsBwd = int(op * (1+op) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeightsBwd)
        featNames.append('fft-weights-bwd')

        # fft-sum-1
        varList.append(fftIfmBwd + fftOfm)
        featNames.append('fft-sum-ifm-bwd-ofm')
        
        # fft-sum-2
        varList.append(fftIfm + fftWeights)
        featNames.append('fft-sum-ifm-weights')
        
        # fft-sum-3
        varList.append(fftOfm + fftWeightsBwd)
        featNames.append('fft-sum-ofm-weights-bwd')

        # fft-sum-all
        varList.append(fftIfm + fftIfmBwd + fftOfm + fftWeights + fftWeightsBwd)
        featNames.append('fft-sum-all')
        
        ###### Winograd implementation 
        # winograd forward
        m = op 
        r = config['k'] 
        winoFwd = int(effIc * (m+r-1)**2)
        varList.append(winoFwd)
        featNames.append('wino-fwd')
        
        # winograd tiles grad wrt inputs 
        m = config['ip']
        r = config['k']
        winoIfmGrad = int(config['oc'] * (m+r-1)**2)
        varList.append(winoIfmGrad)
        featNames.append('wino-gradIfm')

        # winograd tiles grad wrt weights
        m = config['k'] 
        r = config['ip']
        winoWeightGrad = int((m+r-1)**2)
        varList.append(winoWeightGrad)
        featNames.append('wino-gradWeight')

        # winograd sum-1
        varList.append(winoFwd + winoIfmGrad)
        featNames.append('wino-sum-1')
        
        # winograd sum-2
        varList.append(winoFwd + winoWeightGrad)
        featNames.append('wino-sum-2')
        
        # winograd sum-3
        varList.append(winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-3')
        
        # winograd sum-all
        varList.append(winoFwd + winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-all')
        
        return featNames, varList
    #}}}
    
    @classmethod
    def model_v4(cls, config):
    #{{{
        varList = []
        featNames = []
        
        op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
        config['op'] = op
        effIc = (int(config['ic'])/int(config['groups']))
        config['effIc'] = effIc

        ###### GEMM ops 
        gemmOps = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
        varList.append(gemmOps)
        featNames.append('gemm-ops')

        ###### fft ops
        fftFwd = 2 * (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        varList.append(fftFwd)
        featNames.append('fft-fwd-ops')
        
        fftBwdIfm = (op**2 * math.log(op,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * op**2)
        varList.append(fftBwdIfm)
        featNames.append('fft-bwd-ifm-ops')

        ###### winograd ops
        m = op 
        r = config['k'] 
        opsFwd = effIc * (m+r-1) * r**2
        varList.append(opsFwd)
        featNames.append('wino-fwd-ops')
        
        m = config['ip']
        r = config['k']
        opsBwdIfm = config['oc'] * (m+r-1) * r**2
        varList.append(opsBwdIfm)
        featNames.append('wino-bwd-ifm-ops')
        
        m = config['k'] 
        r = config['ip']
        opsBwdWeights = (m+r-1) * r**2
        varList.append(opsBwdWeights)
        featNames.append('wino-bwd-weights-ops')

        ###### tensors allocations 
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
        
        ###### GEMM implementation 
        # i2c-indices
        i2cIdx = int(config['bs'] * op**2)
        varList.append(i2cIdx)
        featNames.append('gemm-i2c-idx')
        
        # i2c-total
        i2c = int(config['bs'] * op**2 * config['k']**2 * config['ic'])
        varList.append(i2c)
        featNames.append('gemm-i2c')

        ###### FFT implementation 
        # fft ifm 
        fftIfm = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['ic'])) 
        varList.append(fftIfm)
        featNames.append('fft-ifm')
        
        # fft ifm 
        fftIfmBwd = int(op * (1+op) * (config['bs'] * config['ic'])) 
        varList.append(fftIfmBwd)
        featNames.append('fft-ifm-bwd')
        
        # fft ofm
        fftOfm = int(op * (1+op) * (config['bs'] * config['oc'])) 
        varList.append(fftOfm)
        featNames.append('fft-ofm')
        
        # fft weights
        fftWeights = int(config['ip'] * (1+config['ip']) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeights)
        featNames.append('fft-weights')
        
        # fft weights
        fftWeightsBwd = int(op * (1+op) * (config['oc'] * (int(config['ic'])/int(config['groups']))))
        varList.append(fftWeightsBwd)
        featNames.append('fft-weights-bwd')

        # fft-sum-1
        varList.append(fftIfmBwd + fftOfm)
        featNames.append('fft-sum-ifm-bwd-ofm')
        
        # fft-sum-2
        varList.append(fftIfm + fftWeights)
        featNames.append('fft-sum-ifm-weights')
        
        # fft-sum-3
        varList.append(fftOfm + fftWeightsBwd)
        featNames.append('fft-sum-ofm-weights-bwd')

        # fft-sum-all
        varList.append(fftIfm + fftIfmBwd + fftOfm + fftWeights + fftWeightsBwd)
        featNames.append('fft-sum-all')
        
        ###### Winograd implementation 
        # winograd forward
        m = op 
        r = config['k'] 
        winoFwd = int(effIc * (m+r-1)**2)
        varList.append(winoFwd)
        featNames.append('wino-fwd')
        
        # winograd tiles grad wrt inputs 
        m = config['ip']
        r = config['k']
        winoIfmGrad = int(config['oc'] * (m+r-1)**2)
        varList.append(winoIfmGrad)
        featNames.append('wino-gradIfm')

        # winograd tiles grad wrt weights
        m = config['k'] 
        r = config['ip']
        winoWeightGrad = int((m+r-1)**2)
        varList.append(winoWeightGrad)
        featNames.append('wino-gradWeight')

        # winograd sum-1
        varList.append(winoFwd + winoIfmGrad)
        featNames.append('wino-sum-1')
        
        # winograd sum-2
        varList.append(winoFwd + winoWeightGrad)
        featNames.append('wino-sum-2')
        
        # winograd sum-3
        varList.append(winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-3')
        
        # winograd sum-all
        varList.append(winoFwd + winoIfmGrad + winoWeightGrad)
        featNames.append('wino-sum-all')
        
        return featNames, varList
    #}}}
    
    @classmethod
    def model_v5(cls, config):
    #{{{
        varList = []
        featNames = []
        
        # op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
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
    
    @classmethod
    # v6 onwards are inference models -- has only forward pass terms
    def model_v6(cls, config):
    #{{{
        varList = []
        featNames = []
        
        # op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
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
        # weightGrad = int(config['bs'] * config['oc'] * effIc * config['k']**2)
        # varList.append(weightGrad)
        # featNames.append('tensor-weight_grad')
        
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
        
        # i2c-bwd_x-indices
        # i2cIdxBwdX = int(config['bs'] * config['ip']**2)
        # varList.append(i2cIdxBwdX)
        # featNames.append('gemm-i2c-idx-bwd-x')
        
        # i2c-bwd_x-total
        # i2cBwdX = int(config['bs'] * config['ip']**2 * config['k']**2 * config['ic'])
        # varList.append(i2cBwdX)
        # featNames.append('gemm-i2c-bwd-x')
        
        # i2c-bwd_w-total
        # i2cBwdW = int(config['bs'] * op**2 * config['k']**2 * effIc)
        # varList.append(i2cBwdW)
        # featNames.append('gemm-i2c-bwd-w')

        # varList.append(i2cFwd + i2cBwdX  +i2cBwdW)
        # featNames.append('gemm-i2c-sum-total')
        
        # varList.append(2*i2cIdxFwd + i2cIdxBwdX)
        # featNames.append('gemm-i2c-index-sum-total')
        #}}}
        
        ###### GEMM ops 
        #{{{
        gemmOpsFwd = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
        varList.append(gemmOpsFwd)
        featNames.append('gemm-ops-fwd')
        
        # # gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['op'] 
        # gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['oc'] 
        # varList.append(gemmOpsBwdX)
        # featNames.append('gemm-ops-bwd')

        # gemmOps = 2*gemmOpsFwd + gemmOpsBwdX
        # varList.append(gemmOps)
        # featNames.append('gemm-ops-sum')
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
        # fftOfmBwdW = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['oc'])) 
        # varList.append(fftOfmBwdW)
        # featNames.append('fft-ofm-bwd-w')
        
        # fft weights bwdx
        # fftWeightsBwdX = int(op * (1+op) * config['oc'] * effIc)
        # varList.append(fftWeightsBwdX)
        # featNames.append('fft-weights-bwd-x')

        # fft ofm bwdx
        # fftOfmBwdX = int(op * (1+op) * (config['bs'] * config['oc'])) 
        # varList.append(fftOfmBwdX)
        # featNames.append('fft-ofm-bwd-x')
        
        # fft-sum-1
        varList.append(fftIfmFwd + fftWeightsFwd)
        featNames.append('fft-sum-fwd')
        
        # fft-sum-2
        # varList.append(fftOfmBwdX + fftWeightsBwdX)
        # featNames.append('fft-sum-bwd_x')
        
        # fft-sum-3
        # fftIfmBwdW = fftIfmFwd
        # varList.append(fftOfmBwdW + fftIfmBwdW)
        # featNames.append('fft-sum-bwd_w')

        # fft-sum-all
        # varList.append(fftIfmFwd + fftWeightsFwd + fftOfmBwdX + fftWeightsBwdX + fftOfmBwdW + fftIfmBwdW)
        # featNames.append('fft-sum-all')
        #}}}
        
        ###### FFT ops
        #{{{
        fftFwdOps = (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        varList.append(fftFwdOps)
        featNames.append('fft-fwd-ops')
        
        # fftBwdXOps = (op**2 * math.log(op,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
        #         config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * op**2)
        # varList.append(fftBwdXOps)
        # featNames.append('fft-bwd_x-ops')
        
        # fftBwdWOps = (config['ip'] * math.log(config['ip']**2,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
        #         config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        # varList.append(fftBwdWOps)
        # featNames.append('fft-bwd_w-ops')

        # varList.append(fftFwdOps + fftBwdXOps + fftBwdWOps)
        # featNames.append('fft-ops-sum')
        #}}}
        
        ###### Winograd implementation 
        #{{{
        def wino_mem(m,r):
        #{{{
            # winograd forward
            fwd = int(3 * config['bs'] * config['oc'] * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
            # winograd tiles grad wrt inputs 
            # bwdX = int(3 * config['bs'] * config['ic'] * (m+r-1)**2 * math.ceil(op/m)**2)
            # winograd tiles grad wrt weights
            # bwdW = int(3 * config['bs'] * config['oc'] * effIc * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
            
            varList.append(fwd)
            featNames.append(f'wino-fwd-{m}-{r}')
            # varList.append(bwdX)
            # featNames.append(f'wino-bwd_x-{m}-{r}')
            # varList.append(bwdW)
            # featNames.append(f'wino-bwd_w-{m}-{r}')
            
            # winograd sum-1
            ## varList.append(fwd + bwdX)
            ## featNames.append(f'wino-sum-1-{m}-{r}')
            # winograd sum-2
            # varList.append(fwd + bwdW)
            # featNames.append('wino-sum-2-{m}-{r}')
            # winograd sum-3
            # varList.append(bwdX + bwdW)
            # featNames.append('wino-sum-3-{m}-{r}')
            # winograd sum-all
            # varList.append(fwd + bwdX + bwdW)
            # featNames.append('wino-sum-all-{m}-{r}')
        #}}}
        
        wino_mem(4,3)
        wino_mem(3,2)
        #}}}

        ###### Winograd ops
        #{{{
        def wino_ops(m,r): 
        #{{{
            fwd = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
            # bwdX = config['bs'] * config['ic'] * config['oc'] * math.ceil(op/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
            # bwdW = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['op']/r)**2 * (m+r-1)**2
            
            featNames.append(f'wino-ops-fwd-{m}-{r}')
            varList.append(fwd)
            # featNames.append(f'wino-ops-bwd_x-{m}-{r}')
            # varList.append(bwdX)
            # featNames.append(f'wino-ops-bwd_w-{m}-{r}')
            # varList.append(bwdW)
            # featNames.append(f'wino-ops-sum-1-{m}-{r}')
            # varList.append(fwd + bwdX)
            # featNames.append(f'wino-ops-sum-2-{m}-{r}')
            # varList.append(fwd + bwdW)
            # featNames.append(f'wino-ops-sum-3-{m}-{r}')
            # varList.append(bwdX + bwdW)
            # featNames.append(f'wino-ops-sum-all-{m}-{r}')
            # varList.append(fwd + bwdX + bwdW)
        #}}}
        
        wino_ops(4,3)
        wino_ops(3,2)
        #}}}
        
        return featNames, varList
    #}}}
    
    @classmethod
    def model_v7(cls, config):
    #{{{
        varList = []
        featNames = []
        
        # op = math.ceil((config['ip'] + 2*config['pad'] - config['k']) / config['stride'])
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
        # weightGrad = int(config['bs'] * config['oc'] * effIc * config['k']**2)
        # varList.append(weightGrad)
        # featNames.append('tensor-weight_grad')
        
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
        
        # i2c-bwd_x-indices
        # i2cIdxBwdX = int(config['bs'] * config['ip']**2)
        # varList.append(i2cIdxBwdX)
        # featNames.append('gemm-i2c-idx-bwd-x')
        
        # i2c-bwd_x-total
        # i2cBwdX = int(config['bs'] * config['ip']**2 * config['k']**2 * config['ic'])
        # varList.append(i2cBwdX)
        # featNames.append('gemm-i2c-bwd-x')
        
        # i2c-bwd_w-total
        # i2cBwdW = int(config['bs'] * op**2 * config['k']**2 * effIc)
        # varList.append(i2cBwdW)
        # featNames.append('gemm-i2c-bwd-w')

        # varList.append(i2cFwd + i2cBwdX  +i2cBwdW)
        # featNames.append('gemm-i2c-sum-total')
        
        # varList.append(2*i2cIdxFwd + i2cIdxBwdX)
        # featNames.append('gemm-i2c-index-sum-total')
        #}}}
        
        ###### GEMM ops 
        #{{{
        gemmOpsFwd = config['bs'] * config['oc'] * op**2 * config['k']**2 * effIc
        varList.append(gemmOpsFwd)
        featNames.append('gemm-ops-fwd')
        
        # # gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['op'] 
        # gemmOpsBwdX = config['bs'] * config['ic'] * config['ip']**2 * config['k']**2 * config['oc'] 
        # varList.append(gemmOpsBwdX)
        # featNames.append('gemm-ops-bwd')

        # gemmOps = 2*gemmOpsFwd + gemmOpsBwdX
        # varList.append(gemmOps)
        # featNames.append('gemm-ops-sum')
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
        # fftOfmBwdW = int(config['ip'] * (1+config['ip']) * (config['bs'] * config['oc'])) 
        # varList.append(fftOfmBwdW)
        # featNames.append('fft-ofm-bwd-w')
        
        # fft weights bwdx
        # fftWeightsBwdX = int(op * (1+op) * config['oc'] * effIc)
        # varList.append(fftWeightsBwdX)
        # featNames.append('fft-weights-bwd-x')

        # fft ofm bwdx
        # fftOfmBwdX = int(op * (1+op) * (config['bs'] * config['oc'])) 
        # varList.append(fftOfmBwdX)
        # featNames.append('fft-ofm-bwd-x')
        
        # fft-sum-1
        varList.append(fftIfmFwd + fftWeightsFwd)
        featNames.append('fft-sum-fwd')
        
        # fft-sum-2
        # varList.append(fftOfmBwdX + fftWeightsBwdX)
        # featNames.append('fft-sum-bwd_x')
        
        # fft-sum-3
        # fftIfmBwdW = fftIfmFwd
        # varList.append(fftOfmBwdW + fftIfmBwdW)
        # featNames.append('fft-sum-bwd_w')

        # fft-sum-all
        # varList.append(fftIfmFwd + fftWeightsFwd + fftOfmBwdX + fftWeightsBwdX + fftOfmBwdW + fftIfmBwdW)
        # featNames.append('fft-sum-all')
        #}}}
        
        ###### FFT ops
        #{{{
        fftFwdOps = (config['ip']**2 * math.log(config['ip'],2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
                config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        varList.append(fftFwdOps)
        featNames.append('fft-fwd-ops')
        
        # fftBwdXOps = (op**2 * math.log(op,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
        #         config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * op**2)
        # varList.append(fftBwdXOps)
        # featNames.append('fft-bwd_x-ops')
        
        # fftBwdWOps = (config['ip'] * math.log(config['ip']**2,2) * (config['bs'] * config['ic'] + config['bs'] * config['oc'] + \
        #         config['oc'] * effIc)) + (config['bs'] * config['oc'] * config['ic'] * config['ip']**2)
        # varList.append(fftBwdWOps)
        # featNames.append('fft-bwd_w-ops')

        # varList.append(fftFwdOps + fftBwdXOps + fftBwdWOps)
        # featNames.append('fft-ops-sum')
        #}}}
        
        ###### Winograd implementation 
        #{{{
        def wino_mem(m,r):
        #{{{
            # winograd forward
            fwd = int(3 * config['bs'] * config['oc'] * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
            # winograd tiles grad wrt inputs 
            # bwdX = int(3 * config['bs'] * config['ic'] * (m+r-1)**2 * math.ceil(op/m)**2)
            # winograd tiles grad wrt weights
            # bwdW = int(3 * config['bs'] * config['oc'] * effIc * (m+r-1)**2 * math.ceil(config['ip']/m)**2)
            
            varList.append(fwd)
            featNames.append(f'wino-fwd-{m}-{r}')
            # varList.append(bwdX)
            # featNames.append(f'wino-bwd_x-{m}-{r}')
            # varList.append(bwdW)
            # featNames.append(f'wino-bwd_w-{m}-{r}')
            
            # winograd sum-1
            ## varList.append(fwd + bwdX)
            ## featNames.append(f'wino-sum-1-{m}-{r}')
            # winograd sum-2
            # varList.append(fwd + bwdW)
            # featNames.append('wino-sum-2-{m}-{r}')
            # winograd sum-3
            # varList.append(bwdX + bwdW)
            # featNames.append('wino-sum-3-{m}-{r}')
            # winograd sum-all
            # varList.append(fwd + bwdX + bwdW)
            # featNames.append('wino-sum-all-{m}-{r}')
        #}}}
        
        wino_mem(4,3)
        wino_mem(3,2)
        #}}}

        ###### Winograd ops
        #{{{
        def wino_ops(m,r): 
        #{{{
            fwd = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
            # bwdX = config['bs'] * config['ic'] * config['oc'] * math.ceil(op/m)**2 * math.ceil(config['k']/r)**2 * (m+r-1)**2
            # bwdW = config['bs'] * config['oc'] * effIc * math.ceil(config['ip']/m)**2 * math.ceil(config['op']/r)**2 * (m+r-1)**2
            
            featNames.append(f'wino-ops-fwd-{m}-{r}')
            varList.append(fwd)
            # featNames.append(f'wino-ops-bwd_x-{m}-{r}')
            # varList.append(bwdX)
            # featNames.append(f'wino-ops-bwd_w-{m}-{r}')
            # varList.append(bwdW)
            # featNames.append(f'wino-ops-sum-1-{m}-{r}')
            # varList.append(fwd + bwdX)
            # featNames.append(f'wino-ops-sum-2-{m}-{r}')
            # varList.append(fwd + bwdW)
            # featNames.append(f'wino-ops-sum-3-{m}-{r}')
            # varList.append(bwdX + bwdW)
            # featNames.append(f'wino-ops-sum-all-{m}-{r}')
            # varList.append(fwd + bwdX + bwdW)
        #}}}
        
        wino_ops(4,3)
        wino_ops(3,2)
        #}}}
        
        return featNames, varList
    #}}}
    
class Fitter(): 
    def __init__(self, modelVersion): 
    #{{{
        self.bs = []
        self.modelVersion = modelVersion
        self.modelFunc = eval("BasicConv.model_{}".format(modelVersion))
    #}}}

    def get_fit_variables(self, batchSize, netName, modelFileName, dataset): 
    #{{{
        # file uniquified using last modification time of generated pruned model
        # this way if the model is pruned again, the description file will be changed
        if 'nvidia' in modelFileName: 
            modelFileName = modelFileName.replace('nvidia/','')
        netFile = "{}_{}.ini".format(modelFileName.split('/')[-1].split('.')[0], int(os.path.getmtime(modelFileName)))
        netDescFile = "{}/{}/{}".format(os.path.dirname(os.path.realpath(__file__)), 'layer_desc_imagenet', netFile)
        if not os.path.isfile(netDescFile):
            print(f"Creating layer description {netDescFile}")
            # if 'nvidia' in modelFileName: 
            #     modelFileName = modelFileName.replace('nvidia/','')
            model = utils.read_model(modelFileName, netName) 
            testGenerator = TestGenerator()
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

            if self.modelVersion == 'v3':
                self.featureNames, memPreds = self.modelFunc(config, nonAccVars)
            else:
                self.featureNames, memPreds = self.modelFunc(config)
            
            if i == 0:
                predictions = np.array(memPreds)
            else:
                predictions += memPreds
        
        predictions = list(predictions)
        if self.modelVersion == 'v3':
            for k,v in nonAccVars.items(): 
                self.featureNames.append(f"mean_{k}")
                predictions.append(np.mean(v))
        
        self.bs.append(batchSize)
        return predictions
    #}}}
    
    def decision_tree(self, data): 
    #{{{
        cases = [x[0:-1] for x in data]
        gt = [x[-1] for x in data]
        decTree = tree.DecisionTreeRegressor()
        decTree = decTree.fit(cases, gt)
        print("Score = {:.2f}".format(decTree.score(cases, gt)))

        predictions = decTree.predict(cases)
        trainErr = [abs(predictions[i] - gt[i])/gt[i] for i in range(len(gt))]
        print("Mean training error = {:.2f}".format(np.mean(trainErr)))

        # print("Feature Importances --")
        # featImp = decTree.feature_importances_
        # for i,feat in enumerate(self.featureNames): 
        #     print(feat, featImp[i]) 
        
        # ax = plt.subplots(1,1)
        # ax[1].scatter(self.bs, gt)
        # ax[1].scatter(self.bs, predictions)
        # plt.show()

        # graphData = tree.export_graphviz(decTree, out_file=None,\
        #                             feature_names=self.featureNames, filled=True,\
        #                             rounded=True, special_characters=True)
        # graph = graphviz.Source(graphData)
        # graph.render('/home/ar4414/pytorch_training/src/adapt/performance_prediction/tmp/graph.gv', view=True)
         
        return decTree
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

        # print("Feature Importances --")
        # featImp = decTree.feature_importances_
        # sortedFeatures = [(feat,featImp[i]) for i,feat in enumerate(self.featureNames)]
        # sortedFeatures.sort(key = lambda x: x[1])
        # for name,imp in enumerate(sortedFeatures): 
        #     print(name, imp)
        
        # fig, ax = plt.subplots(1,1)
        # ax.scatter(x=self.bs, y=gt, c='g')
        # plt.show()
        # breakpoint()
        
        return predictions, decTree
    #}}}
    
    def gradient_boosting(self, data): 
    #{{{
        cases = [x[0:-1] for x in data]
        gt = [x[-1] for x in data]
        decTree = GradientBoostingRegressor(n_estimators=300, loss='ls')
        decTree = decTree.fit(cases, gt)
        print("Score = {:.2f}".format(decTree.score(cases, gt)))

        predictions = decTree.predict(cases)
        trainErr = [abs(predictions[i] - gt[i])/gt[i] for i in range(len(gt))]
        print("Mean training error = {:.2f}".format(np.mean(trainErr)))

        return decTree
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
        
        # fig, ax = plt.subplots(1,1)
        # ax.scatter(x=self.bs, y=gt, c='g')
        # ax.scatter(x=self.bs, y=predictions, c='r')
        # plt.show()
        
        # ax = plt.subplots(1,1)
        # ax[1].scatter(self.bs, gt)
        # ax[1].scatter(self.bs, predictions)
        # plt.show()

        return predictions
    #}}}







