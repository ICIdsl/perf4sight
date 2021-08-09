import os
import subprocess

import torch
import torch.nn as nn

class TestGenerator(object): 
#{{{
    def __init__(self): 
        self.layerCount = 0
        self.inputSizes = {}

        # common params
        self.bs = [2**i for i in range(1,9)]
        self.bs.reverse()
        self.ic = [i for i in range(1,513)]
        self.oc = [i for i in range(1,513)]
        self.ip = [(i,i) for i in range(4, 230)]
        
    def forward_hook(self, module, input, output): 
        self.inputSizes[self.layerCount] = input[0].shape 
        self.layerCount += 1

    def from_model(self, model, dataset, modelConfig): 
    #{{{
        """ Generates a layer by layer test given an input model by identifying conv and fc layers """
        hooks = []
        for n,m in model.named_modules(): 
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(self.forward_hook))
        
        batchSize = 8
        ipImgSize = [32,32] if 'cifar' in dataset else [224,224]
        dummyIp = torch.FloatTensor(batchSize, 3, *ipImgSize)
        
        self.layerCount = 0
        model(dummyIp)        
        
        [h.remove() for h in hooks]

        configOpFile = modelConfig 
        allTests = []
        self.layerCount = 0
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                keys = {'ic':'in_channels', 'oc':'out_channels', 'k':'kernel_size', 'pad':'padding', 'stride':'stride', 'groups':'groups'}
                test = ["[BASIC_CONV-{}]".format(n)]  
                test.append("mode=single")
                test.append("bs={}".format(batchSize))
                test.append("bias={}".format(m.bias is not None))
                ipSize = list(self.inputSizes[self.layerCount])[2:]
                test.append("ip={}".format(ipSize))
                for k,v in keys.items():
                    test.append("{}={}".format(k,eval("m.{}".format(v))))
                allTests.append('\n'.join(test))

                self.layerCount += 1
           
        dirName = os.path.dirname(configOpFile)
        cmd = f"mkdir -p {dirName}"
        subprocess.check_call(cmd, shell=True)
        with open(configOpFile, 'w') as f:
            f.write('\n\n'.join(allTests))
    #}}}
#}}}

class RandomTestGenerator(TestGenerator): 
#{{{
    def __init__(self, params):
        super().__init__(params)

    def basic_conv(self): 
    #{{{
        test = ['[BASIC_CONV_0]']
        test.append('mode=list')
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('k={}'.format([1,3,5,7,9,11]))
        test.append('stride={}'.format([(1,1), (2,2), (4,4)]))
        test.append('groups={}'.format([1,-1]))
        test.append('bias={}'.format([True, False]))

        return '\n'.join(test)
    #}}}
    
    def residual(self): 
    #{{{
        test = ['[RESIDUAL_BASIC_0]']
        test.append('mode=list')
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('stride={}'.format([(1,1), (2,2)]))
        
        test.append('\n[RESIDUAL_BOTTLENECK_0]')
        test.append('mode=list')
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('stride={}'.format([(1,1), (2,2)]))

        return '\n'.join(test)
    #}}}
    
    def mbconv(self): 
    #{{{
        test = ['[MBCONV_0]']
        test.append('mode=list')
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('stride={}'.format([(1,1), (2,2)]))
        test.append('expansion={}'.format([1,2,4,6,8,10,12]))
        
        return '\n'.join(test)
    #}}}
    
    def fire(self): 
    #{{{
        test = ['[FIRE_0]']
        test.append('mode=list')
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('sc={}'.format([2,4,8,16,32,42,48,52,64,128]))
        
        return '\n'.join(test)
    #}}}
#}}}

class SingleChangeTestGenerator(TestGenerator):
#{{{
    def __init__(self, params):
        super().__init__(params)
       
        self.bs = [256, 200, 128, 100, 64, 48, 32, 16, 8, 4, 2]
        self.ic = [3, 8, 16, 32, 48, 64, 80, 128, 200, 256]
        self.oc = [16, 20, 32, 48, 64, 80, 128, 200, 256]
        self.ip = [(32,32), (225,225)]

    def basic_conv(self): 
    #{{{
        test = ['[BASIC_CONV_0]']
        test.append('mode=exhaustive')
        
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('k={}'.format([3,5,11]))
        test.append('stride={}'.format([(1,1), (2,2), (4,4)]))
        test.append('groups={}'.format([1]))
        test.append('bias={}'.format([True]))

        return '\n'.join(test)
    #}}}
    
    def residual(self): 
    #{{{
        test = ['[RESIDUAL_BASIC_0]']
        test.append('mode=exhaustive')

        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('stride={}'.format([(1,1), (2,2)]))
        
        # test.append('\n[RESIDUAL_BOTTLENECK_0]')
        # test.append('mode=list')
        # test.append('bs={}'.format(self.bs))
        # test.append('ic={}'.format(self.ic))
        # test.append('oc={}'.format(self.oc))
        # test.append('ip={}'.format(self.ip))
        # 
        # test.append('stride={}'.format([(1,1), (2,2)]))

        return '\n'.join(test)
    #}}}
    
    def mbconv(self): 
    #{{{
        test = ['[MBCONV_0]']
        test.append('mode=exhaustive')
        
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('stride={}'.format([(1,1), (2,2)]))
        test.append('expansion={}'.format([1,6]))
        
        return '\n'.join(test)
    #}}}
    
    def fire(self): 
    #{{{
        test = ['[FIRE_0]']
        test.append('mode=exhaustive')
        
        test.append('bs={}'.format(self.bs))
        test.append('ic={}'.format(self.ic))
        test.append('oc={}'.format(self.oc))
        test.append('ip={}'.format(self.ip))
        
        test.append('sc={}'.format([16,32,48,64]))
        
        return '\n'.join(test)
    #}}}
#}}}
