import numpy as np
from waff.Layers.layer import Layer
from waff.Layers.ops.mat_calucation import Pool2dAverage,Pool2dMax,Pool2dAverageBack,Pool2dMaxBack

class Pool(Layer):
    def __init__(self,kernelsizes,type_,stride = None ,padding=0,name = ''):
        Layer.__init__(self,type_ = type_,name = name)
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
            
    def adding_pad(self,map):
        pp = self.padding
        if self.padding != 0:
            paddingArray=np.pad(map,pad_width=((0,0),(0,0),(pp,pp), (pp,pp)),mode='constant')
        else:
            paddingArray = map
        return paddingArray
    
    
    def passForward(self):
        pass
    
    def backpropagate(self,delta,backLayerDelta=True):
        pass

class MaxPool2d(Pool):
    def __init__(self,kernelsizes,stride = None ,padding=0,name = ''):
        Pool.__init__(self,kernelsizes,'MAXPOOL',stride,padding,name)
    def passForward(self):
        self.paddedMap = self.adding_pad(self.input)
        (self.batch_size,self.inputChannelSize,self.height,self.width)=self.paddedMap.shape
        if self.phase.lower() == 'train':
            self.z_,self.maxBase = Pool2dMax(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride,self.phase.lower())
        else:
            self.z_ =   Pool2dMax(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
        self.a_ =self.z_
    
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            delta_back = Pool2dMaxBack(delta,self.paddedMap,self.maxBase,self.kernelheight,self.kernelwidth,self.stride)
            if self.padding != 0:
                deltaBackLayer = delta_back[:,:,self.padding:-self.padding,self.padding:-self.padding]
            else:
                deltaBackLayer = delta_back
            assert deltaBackLayer.shape == self.input.shape
            return deltaBackLayer
        else:
            return None

class AvgPool2d(Pool):
    def __init__(self,kernelsizes,stride = None ,padding=0,name = ''):
        Pool.__init__(self,kernelsizes,'AVGPOOL',stride,padding,name)
    def passForward(self):

        self.paddedMap = self.adding_pad(self.input)
        (self.batch_size,self.inputChannelSize,self.height,self.width)=self.paddedMap.shape 
        self.a_ = Pool2dAverage(self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
   

    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            delta_back = Pool2dAverageBack(delta,self.paddedMap,self.kernelheight,self.kernelwidth,self.stride)
            if self.padding != 0:
                deltaBackLayer = delta_back[:,:,self.padding:-self.padding,self.padding:-self.padding]
            else:
                deltaBackLayer = delta_back
            assert deltaBackLayer.shape == self.input.shape
            return deltaBackLayer
        else:
            return None
