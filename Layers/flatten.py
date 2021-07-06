import numpy as np
from waff.Layers.layer import Layer



class Flatten(Layer):
    def __init__(self,type_ = 'Flatten',name = ''):
        Layer.__init__(self,type_ = type_, name = name)
        
    def passForward(self):
        assert len(self.input.shape) == 4
        self.a_ = self.input.reshape(self.input.shape[0],-1)
        
    def backpropagate(self,delta,backLayerDelta = True):
        if backLayerDelta:
            deltaBackLayer = delta.reshape(self.input.shape)
        else:
            deltaBackLayer = None
        
        return deltaBackLayer
            
