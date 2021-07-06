import numpy as np
from waff.Layers.layer import Layer
from waff.Layers.ops.mat_calucation import Pool2dAverage,Pool2dMax,Pool2dAverageBack,Pool2dMaxBack

class Pool(Layer):
    def __init__(self,kernelsizes=[2,2],poolType='average',stride = None ,padding=0,name = ''):
        Layer.__init__(self,type_ = 'POOL',name = name)
        self.variableNeedInitialize=True
        self.kernel_(kernelsizes)
        if self.kernelheight != self.kernelwidth:
            raise AssertionError('kernel height should be equal to kernel width')
        if stride is not None and type(stride)==int:
            self.stride = stride
        else:
            self.stride = self.kernelheight
        self.paddedMap = None
        if type(padding) == int:
            self.padding = padding
        else:
            self.padding = 0
        self.poolType=poolType.lower()
        if self.poolType =='mean':
            self.poolType = 'average'
        
    
        if self.poolType !='average' and self.poolType != 'max':
            raise TypeError('No {} defined'.format(self.poolType))
        
            
    def adding_pad(self,map):
        pp = self.padding
        if self.padding != 0:
            paddingArray=np.pad(map,pad_width=((0,0),(0,0),(pp,pp), (pp,pp)),mode='constant')
        else:
            paddingArray = map
        return paddingArray

   
    
    def initiateVariables(self):
        self.out_h = (self.height-self.kernelheight)/self.stride+1
        self.out_w = (self.width-self.kernelwidth)/self.stride+1
        self.effectHeight = self.out_h * self.stride + self.kernelheight
        self.effectWidth = self.out_w * self.stride + self.kernelwidth
        self.variableNeedInitialize=False
    
    
    def passForward(self):
        self.paddedMap = self.adding_pad(self.input)
        (self.batch_size,self.inputChannelSize,self.height,self.width)=self.paddedMap.shape
        if self.variableNeedInitialize:
            self.initiateVariables()
        if self.poolType == 'average':
            self.z_ = Pool2dAverage(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
        elif self.poolType == 'max':
            if self.phase.lower() == 'train':
               self.z_,self.maxBase = Pool2dMax(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride,self.phase.lower())
            else:
                self.z_ = Pool2dAverage(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
        else:
            pass
        self.a_ =self.z_
    
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            
            if self.poolType == 'average':
                delta_back = Pool2dAverageBack(delta,self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
            elif self.poolType == 'max':
                delta_back = Pool2dMaxBack(delta,self.paddedMap,self.maxBase,self.kernelheight,self.kernelwidth,self.stride)
            else:
                pass
            if self.padding != 0:
                deltaBackLayer = delta_back[:,:,self.padding:-self.padding,self.padding:-self.padding]
            else:
                deltaBackLayer = delta_back
            assert deltaBackLayer.shape == self.input.shape
            return deltaBackLayer
        else:
            return False

