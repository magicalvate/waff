import numpy as np
from waff.Layers.layer import Layer
class Dropout(Layer):
    def __init__(self,propotion = 0.5,name = ''):
        Layer.__init__(self,type_ = 'DROPOUT',name = name)
        if propotion < 1 and propotion>=0:
            self.p = propotion
        else:
            raise Exception('propotion must be in range[0,1)')
        
    def passForward(self):
        if self.phase.lower() == 'test':
            self.a_ = self.input
        elif self.phase.lower() == 'train':
            '''
            length = np.product(self.input.shape[:])
            arr = np.ones(length)
            
            index = np.random.choice(np.arange(length),int(length*self.p),replace = False)
            arr[index] = 0
            self.drop_arr = np.tile(arr,self.input.shape[0]).reshape(self.input.shape)
            self.a_ = self.drop_arr*self.input
            self.a_ = self.a_/(1-self.p)
            '''
            drop_arr = np.random.rand(self.input.shape)
            self.drop_arr = np.where(drop_arr<self.p,0,1)
            self.a_ = self.drop_arr*self.input
            self.a_ = self.a_/(1-self.p)

        else:
            raise TypeError('no legal phase defined')
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            deltaBackLayer = delta/(1-self.p)*self.drop_arr
            return deltaBackLayer
        else:
            pass
