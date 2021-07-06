import numpy as np
from waff.Layers.layer import Layer
from waff.Layers.ops.mat_calucation import Matmul

class Dense(Layer):

    def __init__(self,outputnumber,lr_base=[1,2],name = '',kernel_initializer ='GlorotNormal' ,bias_initializer ='Zeros',activation = None):
        Layer.__init__(self,lr_base,'DENSE',name,kernel_initializer,bias_initializer,activation)
        self.outputnumber = outputnumber
          
    def __initiateVariables(self,inputFeatureMaps):
        (self.batch_size,self.inputChannelSize,self.height,self.width) = inputFeatureMaps.shape
        self.inputnumber = self.inputChannelSize*self.height*self.width
        self.initiate_variable(1,1,self.outputnumber,self.inputnumber,self.type)
        self.variableNeedInitialize=False
        if self.inputnumber!=self.height:
            self.backLayerTpye = 'MAP'
        else:
            self.backLayerTpye = 'DENSE'
            
    def __matmul(self,w,a,b = 0):
        return Matmul(w,a,b)
        
    def passForward(self):
        self.batch_size = self.input.shape[0]
        if self.variableNeedInitialize:
            self.__initiateVariables(self.input)
        
        if self.backLayerTpye == 'MAP':
            self.__flattenInput = self.input.reshape(self.batch_size,1,self.inputnumber,1)
        else:
            self.__flattenInput = self.input.copy()
    
        
        self.z_ = self.__matmul(self.w_,self.__flattenInput,self.b_)
        try:
            self.a_ = self.activation(self.z_)
        except:
            self.a_ = self.z_
        
    def __calculateDeltaBackLayer(self,delta):
        value = self.__matmul(self.w_.swapaxes(-1,-2),delta)
        if self.backLayerTpye == 'MAP':
            value = value.reshape(self.batch_size,self.inputChannelSize,self.height,self.width)
        assert value.shape == (self.batch_size,self.inputChannelSize,self.height,self.width)
        return value
        
    def backpropagate(self,delta,backLayerDelta=True):
        try:
            delta = self.activation.backpropagate(delta)
        except:
            pass
        if backLayerDelta:
            deltaBackLayer = self.__calculateDeltaBackLayer(delta)
        else:
            deltaBackLayer = None
        delta_w_ = self.__matmul(delta,self.__flattenInput.swapaxes(-1,-2))
        assert delta_w_.shape == self.w_.shape
        delta_b_= np.sum(delta,axis = 0)[np.newaxis,...]
        assert delta_b_.shape == self.b_.shape
        delta_w_ /= self.batch_size
        delta_b_ /= self.batch_size
        return deltaBackLayer,delta_w_,delta_b_
