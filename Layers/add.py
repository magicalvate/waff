import numpy as np
from waff.Layers.layer import Layer
class Add(Layer):
    def __init__(self,*mix_layers):
        Layer.__init__(self,type_ = 'ADD',name = '')
        self.mix_layers = mix_layers
        self.sub_num = len(mix_layers)

        if self.sub_num<2:
            raise Exception('Input must be greater than or equal to 2')
        for i in range(self.sub_num):
            if issubclass(type(mix_layers[i]),Layer):
                pass
            else:
                raise TypeError('Input must be subclass of Layer')
            
        
    def passForward(self):
        for i,layer in enumerate(self.mix_layers):
            
            if i == 0:
                self.a_ = layer.a_
            else:
                assert self.a_.shape == layer.a_.shape
                self.a_ += layer.a_

       
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            deltaBackLayer = []
            for i,el in enumerate(self.mix_layers):
                assert delta.shape == self.mix_layers[i].a_.shape
                deltaBackLayer.append(delta)
            return deltaBackLayer
        else:
            pass
