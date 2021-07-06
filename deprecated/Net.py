# -*- coding: utf-8 -*-
import os,struct
import random
import numpy as np
from numpy import append, array, int8, uint8, zeros
import pickle
import gzip
#import waff.activation
from waff.layers import *
from waff.optimizer import *

from waff.loss import  Loss
from waff.utils import *
import time


# In[2]:


class Net(object):
    def __init__(self,layerlist,loss_type = 'mse'):
        self.layerlist=layerlist
        self.namelist = []
        self.typelist ={}
        for layer in self.layerlist:
            layer.name = create_layername(layer.name,layer.type,self.namelist)
            self.namelist.append(layer.name)
            self.typelist.update({layer.name:layer.type})
        self.loss=0
        self.netOutput=0
        try:
            lastActivation = self.layerlist[-1].activation.method.upper()
        except:
            lastActivation = self.layerlist[-1].type.upper()
        loss_type = self.loss_type(loss_type.upper(),lastActivation.upper())
        print ('loss for net',loss_type)
        self.loss_calculator = Loss(ltype = loss_type)
    def __call__(self,X,phase = 'Train'):
        self.passForward(X,phase)
        return self.netOutput
        
    def loss_type(self,loss_type,activation):
        if loss_type == 'LOGLOSS':
            if activation == 'SOFTMAX':
                loss_type = 'SOFTMAXWITHLOSS'
            return loss_type
        else:
            return loss_type
    def intialize_optimizer(self,optimizer):
        self.copy_variable_to_optimizer = True
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = SGD()
        self.optimizer.namelist = self.namelist
        
    def train(self,train_data,epochs,mini_batch_size,test_epoch=20,print_iteration=20,rate_decay=0.001,weight_decay=0.0005,test_data=None,optimizer = None):
        self.intialize_optimizer(optimizer)
        
        self.optimizer.weight_decay = weight_decay
        if test_data:
            n_test=len(test_data)
        n=len(train_data)
        #print '******************length of train_data:',n
        number_iter_per_epoch = len(range(0, n, mini_batch_size))
        print ('number_iter_per_epoch is :',number_iter_per_epoch)
        for i in range(epochs):
            predict_train = []
            gt = []
            self.loss = 0
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            if i%test_epoch==0:
                    test_start=time.time()
                    if test_data:
                        loss_a,accurate = self.evaluate(test_data)
                        print("Epoch {} test result: {} / {}, loss:{:.3f}".format(i, accurate, n_test,loss_a))
                        print("time used for testing is {:.3f}second".format(time.time()-test_start))
                    else:
                        print("Epoch {} complete".format(i))

            start_time=time.time()
            for j,mini_batch in enumerate(mini_batches):
                try:
                    self.optimizer.lr *= (1/(1+rate_decay*(i*number_iter_per_epoch+j)))
                except:
                    pass
                (X,y)=zip(*mini_batch)
                predict_y =  self(np.array(X),'Train').tolist()
                predict_train.extend(predict_y)
                gt.extend(y.tolist())
                self.y = y
                for layer in self.layerlist:
                    try:
                        self.optimizer.lr_w_.update({layer.name:self.optimizer.lr * layer.lr_base_weight})
                        self.optimizer.lr_b_.update({layer.name:self.optimizer.lr * layer.lr_base_bias})
                    except:
                        pass
                      
                
                self.backpropagate(y)
                self.calculateLoss(np.array(y))
                if (j+1)%print_iteration==0:
                    self.loss = self.loss/print_iteration
                    test_results = [(np.argmax(predict_train[i]), np.argmax(y)) for i,(x, y) in enumerate(test_data)]
                    print("Epoch {} iteration{} completed with Loss:{:.3f}, trainning process cost:{:.3f} second".\
                          format(i,j,self.loss,time.time()-start_time))
                    print("Epoch {} iteration{} completed with accuracy{}/{}".format(,len(predict_train)))
                    start_time=time.time()
                    self.loss = 0
                    #print('output calculated{0},y{1}'.format(self.netOutput[0],y[0]))

                
                
    def evaluate(self,test_data,batch_size = 40):
        predict_ = []
        y_ = []
        mini_batches = [test_data[k:k+batch_size] for k in range(0, len(test_data), batch_size)]
        for j,mini_batch in enumerate(mini_batches):
            (X,y)=zip(*mini_batch)
            #test_results0 = [(self.passForward(np.array([x])), np.array([y])) for (x, y) in test_data]
            predict = self(np.array(X),'Test').tolist()
            predict_.extend(predict)
            y_.extend(y)
        loss_ev = self.loss_calculator.calculateLoss(np.array(predict_),np.array(y_))
        test_results = [(np.argmax(predict_[i]), np.argmax(y)) for i,(x, y) in enumerate(test_data)]
        
        return loss_ev,sum(int(x == y) for (x, y) in test_results)

    
    def passForward(self,batchData,phase):
        for layer in self.layerlist:
            layer.phase = phase
            layer.passForward(batchData)
            batchData=layer.a_
            if self.copy_variable_to_optimizer:
                try:
                    self.optimizer.w_.update({layer.name:layer.w_})
                    self.optimizer.b_.update({layer.name:layer.b_})
                    if layer.type == 'BATCHNORM':
                        self.optimizer.bn_param.update({layer.name:layer.bn_param})
                except:
                    print("layer {} doesn't have W and b value ".format(layer.name))
        self.copy_variable_to_optimizer = False
        self.netOutput=batchData
        
    
        
    def calculateLoss(self,y):
        self.loss += self.loss_calculator.calculateLoss(self.netOutput,y)
    
    def backpropagate(self,y):
        #for i in range(1,len(self.layerlist)+1):
        n=len(self.layerlist)
        #print ('length of net sequential is {}'.format(n))

        for i in range(1,n+1):
            layer=self.layerlist[-i]
            name_curr = self.namelist[-i]
            if i == 1:
                delta = self.loss_calculator.backPropagate(self.netOutput,y).copy()
                self.optimizer.delta.update({name_curr:delta})
            else:
                delta = self.optimizer.delta[name_curr]
                #print "current layer is lat layer"
            
            if i != n:
                delta_return  = layer.backpropagate(delta)
            else:
                delta_return= layer.backpropagate(delta,backLayerDelta=False)
            try:
                delta_back,delta_w_,delta_b_ = delta_return
                self.optimizer.delta_w_.update({layer.name:delta_w_})
                self.optimizer.delta_b_.update({layer.name:delta_b_})
            except:
                delta_back = delta_return
            #last layer don't need to update delta for previous layer
            if i!= n:
                name_prev = self.namelist[-(i+1)]
                self.optimizer.delta.update({name_prev:delta_back})
        self.optimizer.update()
        for layer in self.layerlist:
            try:
                layer.w_ = self.optimizer.w_[layer.name].copy()
                layer.b_ = self.optimizer.b_[layer.name].copy()
            except:
                pass
           









