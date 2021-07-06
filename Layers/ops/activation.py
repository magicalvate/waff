# coding: utf-8
import numpy as np
import time
#激活函数
class Activation(object):
    def __init__(self,method = 'sigmoid'):
        if method.lower() in ['sigmoid','tanh','softmax','relu','none','leakyrelu']:
            self.method = method.lower()
            self.input = 0.0
            self.a_ = 0.0
        else:
            raise TypeError("{} can't be found ,should be one of listed below,\n'none','sigmoid','tanh','softmax','relu','leakyrelu' ".format(method))
    def __call__(self,z):
        self.passForward(z)
        return self.a_
        
    def passForward(self,z):
        self.input = z
        self.a_ = eval('self.'+self.method+'(z)')
        #print self.a_
        #print np.sum(self.a_)
        
    def backpropagate(self,delta,backLayerDelta=True):
        if backLayerDelta:
            deltaBackLayer = delta*eval('self.'+self.method+'dev(self.input)')
            return deltaBackLayer
        else:
            return False

    def relu(self,z):
        z1=np.where(z<0,0,z)
        
        return z1
       
    def leakyrelu(self,z):
        z1=np.where(z<0,0.1*z,z)
        return z1
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def tanh(self,z):
    
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        
    def softmax(self,z):
        '''
        z = z.squeeze((1,-1))
        a = np.exp(z)
        b = np.sum(a,axis = -1)[:,np.newaxis]
        b = np.tile(b,[1,z.shape[-1]])
        a = a/b
        a = a[:,np.newaxis,:,np.newaxis]
        '''
        #length = z.shape[-1]
        b = np.exp(z)
        c = np.sum(b,axis = 1,keepdims=True)
        #c = np.tile(c,[1,1,length,1])
        a = b/c
        return a
        
    def none(self,z):
        return z

    def reludev(self,z):
        return np.where(z<0,0,1).astype(z.dtype)

    def leakyreludev(self,z):
        return np.where(z<0,0.1,1).astype(z.dtype)

    def sigmoiddev(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def tanhdev(self,z):
        return 1-self.tanh(z)**2
    def softmaxdev(self,z):
        return np.ones_like(z)
        
    def nonedev(self,z):
        return np.ones_like(z)




