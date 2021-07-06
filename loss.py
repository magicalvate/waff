import numpy as np
from numpy import log as log
import os
class Loss(object):
    def __init__(self,ltype = 'mse'):
        self.ltype = ltype.upper()
        loss = 0
        
    def calculateLoss(self,a_,y):
        assert y.shape == a_.shape
        if self.ltype == 'MSE':
            diff_arr = (a_ - y).squeeze()
            loss= np.sum(diff_arr**2)/a_.shape[0]
            return loss
        elif self.ltype == 'LOGLOSS' or self.ltype =='LOG':
            #print a_.shape
           
            diff_arr = ((y*log(a_+1e-8))+(1-y)*log(1-a_+1e-8)).squeeze()
            diff_arr = diff_arr*-1
            loss = np.sum(diff_arr)/a_.shape[0]
            return loss

        elif self.ltype == 'SOFTMAXWITHLOSS':
           
            diff_arr = (y*log(a_+1e-8)).squeeze()
            diff_arr = diff_arr * -1
            loss = np.sum(diff_arr)/a_.shape[0]
            return loss

        else:
            raise Exception('Loss type {} is not found'.format(self.ltype))
            
    
    def backPropagate(self,a_,y):
        if self.ltype == 'MSE':
            delta = 2*(a_-y)
        elif self.ltype == 'LOGLOSS' or self.ltype =='LOG':
            delta = (1-y)/(1-a_+1e-8)-y/(a_+1e-8)
        elif self.ltype == 'SOFTMAXWITHLOSS':
            delta = (a_-y)
        else:
            raise Exception('Loss type {} is not found'.format(self.ltype))
        return delta
