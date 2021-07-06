import numpy as np
from waff.Layers.layer import Layer
from waff.Layers.ops.mat_calucation import Matmul
class Dense(Layer):

    def __init__(self,out_dim,lr_base=[1,2],name = '',kernel_initializer ='GlorotNormal' ,bias_initializer ='Zeros',activation = None):
        Layer.__init__(self,lr_base,'DENSE',name,kernel_initializer,bias_initializer,activation)
        self.out_dim = out_dim
          

            
    def passForward(self):
        assert len(self.input.shape) == 2
        if self.variableNeedInitialize:
            self.in_dim = self.input.shape[1]
            self.variableNeedInitialize=False
            self.initiate_variable(self.out_dim,self.in_dim)
        self.a_ = np.dot(self.input,self.w_.transpose())+self.b_
        try:
            self.a_ = self.activation(self.a_)
        except:
            pass
        
    def backpropagate(self,delta,backLayerDelta=True):
        try:
            delta = self.activation.backpropagate(delta)
        except:
            pass
        if backLayerDelta:
            deltaBackLayer = Matmul(self.w_,delta,phase = 1)
        else:
            deltaBackLayer = None
        batch_size = delta.shape[0]
        delta_w_ = Matmul(delta,self.input)
        assert delta_w_.shape == self.w_.shape
        delta_b_= np.sum(delta,axis = 0)
        assert delta_b_.shape == self.b_.shape
        delta_w_ /= batch_size
        delta_b_ /= batch_size
        '''
        try:
            print deltaBackLayer.shape
            print delta.shape,self.w_.shape
        except:
            pass
        '''
        return deltaBackLayer,delta_w_,delta_b_
