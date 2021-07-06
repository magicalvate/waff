import numpy as np


#returning the gradient
class Optimizer(object):
    def __init__(self,lr,type_,drop_out = 0.0):
        self.namelist = []
        self.type = type_
        if lr is not None:
            self.lr = lr
            self.lr_w_ = {}
            self.lr_b_ = {}
        self.w_ = {}
        self.b_ = {}
        self.bn_param = {}
        self.dataFlow = {}
        self.drop_out = drop_out
        #gradient for layer
        self.delta = {}
        #self.gradient = {}
        #gradient for weights
        self.delta_w_ = {}
        #gradient for bias
        self.delta_b_ = {}
        self.weight_decay = 0.0
    def update(self):
        self.update_w()
        self.update_b()
        #self.update_gamma()
        #self.update_beta()
    def update_w(self):
        pass
        
    def update_b(self):
        pass
