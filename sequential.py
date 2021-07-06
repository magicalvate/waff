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
from waff.dataFlow import Flow
from waff.loss import  Loss
from waff.Layers.ops.utils import *
import time

class Sequential(object):
    def __init__(self,layerlist = [],loss_type = 'mse'):
        self.layerlist=layerlist
        self.namelist =[]
        self.__typelist ={}
        self.__dicLayer = {}
        for layer in self.layerlist:
            layer.name = create_layername(layer.name,layer.type,self.namelist)
            self.namelist.append(layer.name)
            self.__dicLayer.update({layer.name:layer})
            self.__typelist.update({layer.name:layer.type})
        self.loss=0
        self.output=0
        #self.phase = Train
        try:
            lastActivation = self.layerlist[-1].activation.method.upper()
        except:
            lastActivation = self.layerlist[-1].type.upper()
        loss_type = self.__loss_type(loss_type.upper(),lastActivation.upper())
        print ('loss for net',loss_type)
        self.loss_calculator = Loss(ltype = loss_type)
    def __call__(self,X,phase = 'Test'):
        self.inFlow = Flow(np.array(X).astype('float32'),'data')
        self.__passForward(self.inFlow,phase)
        return self.output
        
    def __loss_type(self,loss_type,activation):
        if loss_type == 'LOGLOSS':
            if activation == 'SOFTMAX':
                loss_type = 'SOFTMAXWITHLOSS'
            return loss_type
        else:
            return loss_type
    def __intialize_optimizer(self,optimizer):
        self.copy_variable_to_optimizer = True
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = SGD()
        self.optimizer.namelist = self.namelist
        
    def train(self,train_data,epochs,batch_size,test_epoch=20,print_iteration=20,rate_decay=0.001,weight_decay=0.0005,test_data=None,optimizer = None,test_batch_size = 40):
        self.__intialize_optimizer(optimizer)
        
        self.optimizer.weight_decay = weight_decay
        for layer in self.layerlist:
            layer.optimizer = self.optimizer
        if test_data:
            n_test=len(test_data)
            test_batches = [test_data[k:k+test_batch_size] for k in range(0, n_test, test_batch_size)]
        n=len(train_data)
        #print '******************length of train_data:',n
        number_iter_per_epoch = len(range(0, n, batch_size))
        print ('number_iter_per_epoch is :',number_iter_per_epoch)
        for i in range(epochs):
            #test_process
            predict_train = []
            gt_train = []
            self.loss = 0
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            if i%test_epoch==0:
                predict_test = []
                gt_test = []
                test_start=time.time()
                if test_data:
                    for j,test_batch in enumerate(test_batches):
                        (X,y)=zip(*test_batch)
                        predict = self(X,'Test').tolist()
                        predict_test.extend(predict)
                        gt_test.extend(y)
                    #print len(predict_test)
                    loss_a,accurate = self.evaluate(predict_test,gt_test)
                    print("Epoch {} test acc: {:.3f}, loss:{:.3f}".format(i, accurate*1.0/n_test,loss_a))
                    print("time used for testing is {:.3f}s".format(time.time()-test_start))
                else:
                    print("Epoch {} complete".format(i))

            start_time=time.time()
            c0 = 0.0
            c1 = 0.0
            for j,mini_batch in enumerate(mini_batches):
                (X,y)=zip(*mini_batch)
                c11 = time.time()
                predict =  self(X,'Train').tolist()
                c0 += (time.time()-c11)
                
                predict_train.extend(predict)
                gt_train.extend(y)
                self.y = y
                try:
                    self.optimizer.lr *= (1/(1+rate_decay*(i*number_iter_per_epoch+j)))
                except:
                    pass
                for layer in self.layerlist:
                    try:
                        self.optimizer.lr_w_.update({layer.name:self.optimizer.lr * layer.lr_base_weight})
                        self.optimizer.lr_b_.update({layer.name:self.optimizer.lr * layer.lr_base_bias})
                    except:
                        pass
                      
                c_11 = time.time()
                #print 'get y shape',y.shape
                self.__backpropagate(y)
                c1 += (time.time()-c_11)
                if (j+1)%print_iteration==0:
                    #print 'fwd',c0
                    #print 'back',c1
                    c0 = 0.0
                    c1 = 0.0
                    #print self.layerlist[2].bn_param
                    loss_a,accurate = self.evaluate(predict_train,gt_train)
                    print("Epoch {} iteration{} completed with Acc:{:.3f} Loss:{:.3f}, process cost:{:.3f}s".\
                          format(i,j+1,accurate*1.0/len(predict_train),loss_a,time.time()-start_time))
                    print("{:.2f} iterations per second".format(print_iteration/(time.time()-start_time)))
                    #print("Epoch {} iteration{} completed with accuracy:{:.3f}".format(i,j+1,))
                    start_time=time.time()
                    '''
                    predict_train = []
                    gt_train = []
                    '''
                    #print('output calculated{0},y{1}'.format(self.output[0],y[0]))

                
                
    def evaluate(self,predict_,y_):
       
        loss_ev = self.loss_calculator.calculateLoss(np.array(predict_),np.array(y_))
        acc_list = [(np.argmax(predict_[i]), np.argmax(y)) for i,y in enumerate(y_)]
        return loss_ev,sum(int(x == y) for (x, y) in acc_list)

    
    def __passForward(self,batchData,phase):
        for layer in self.layerlist:
            layer.phase = phase
            batchData=layer(batchData)
          
    
        self.outFlow=batchData
        self.output = self.outFlow.numpy()
        
    

    
    def __backpropagate(self,y):
        #for i in range(1,len(self.layerlist)+1):
        n=len(self.layerlist)
        self.optimizer.delta = {}
        cur_layer = self.namelist[-1]
        prior_layer = cur_layer
        previous_layer = ''
        while previous_layer != 'data':
            layer = self.__dicLayer.get(cur_layer)
            previous_layer = self.optimizer.dataFlow.get(cur_layer)
            delta = self.optimizer.delta.get(prior_layer,self.loss_calculator.backPropagate(self.output,np.array(y).astype('float32')).copy())
            if previous_layer == 'data':
                layer.back(delta,False)
            else:
                layer.back(delta,True)
            prior_layer = cur_layer
            cur_layer = previous_layer
            
        self.optimizer.update()
        
        for layer in self.layerlist:
            try:
                layer.w_ = self.optimizer.w_[layer.name]
                layer.b_ = self.optimizer.b_[layer.name]
            except:
                pass
        
    def load_weights(self,load_path = 'model_weights.pkl' ):
        if not os.path.exists(load_path):
            print('load path does not exist')
        else:
            pkl_file = open(load_path,'rb')
            u = pickle._Unpickler(pkl_file)
            u.encoding = 'latin1'
            varibles = u.load()

            #varibles = pickle.load(pkl_file)
            for layer in self.layerlist:
                cur_layer_name = layer.name
                try:
                    layer.w_ = varibles['weights'][cur_layer_name]
                    layer.b_ = varibles['bias'][cur_layer_name]
                except:
                    pass
                if layer.type == 'BATCHNORM':
                    try:
                        layer.bn_param = varibles['bn_variables'][cur_layer_name]
                    except:
                        pass
                    
    def save_weights(self,save_path = 'model_weights.pkl'):
        while os.path.exists(save_path):
            elements = os.path.splitext(save_path)
            name = elements[0]
            name += ('_'+time.strftime("%m%d%H%M%S") )
            name += elements[1]
            save_path = name
        pickle_file = open(save_path,'wb')
        varibles = {}
        varibles.update({'weights':self.optimizer.w_})
        varibles.update({'bias':self.optimizer.b_})
        varibles.update({'namelist':self.optimizer.namelist})
        varibles.update({'bn_variables':self.optimizer.bn_param})
        pickle.dump(varibles,pickle_file)
        pickle_file.close()





