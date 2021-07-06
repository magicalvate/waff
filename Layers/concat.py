import numpy as np
from waff.Layers.layer import Layer
class Concat(Layer):
    def __init__(self,*concat_layers):
        Layer.__init__(self,type_ = 'CONCAT',name = '')
        self.concat_layers = concat_layers
        self.sub_num = len(concat_layers)

        if self.sub_num<2:
            raise Exception('Input must be greater than or equal to 2')
       
        for i in range(self.sub_num):
            if issubclass(type(self.concat_layers[i]),Layer):
                pass
            else:
                raise TypeError('Input must be subclass of Layer')
            
        
    def passForward(self):
        self.dimProp  = [0]
       
        for i,layer in enumerate(self.concat_layers):
            if i == 0:
                self.a_ = layer.a_
            else:
                self.a_ = np.concatenate((self.a_,layer.a_),axis = 1)
            total = self.dimProp[-1]
            total += layer.a_.shape[1]
            self.dimProp.append(total)
        #print self.dimProp
       
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            deltaBackLayer = []
            for i,el in enumerate(self.dimProp[:-1]):
                #print el,self.dimProp[i+1]
                delta_pro = delta[:,el:self.dimProp[i+1],...]
                assert delta_pro.shape == self.concat_layers[i].a_.shape
                deltaBackLayer.append(delta_pro)
            return deltaBackLayer
        else:
            pass
