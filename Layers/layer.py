import numpy as np
from waff.Layers.ops import initializers
from waff.Layers.ops.activation import Activation
from waff.dataFlow import Flow
import time
class Layer(object):
    #inputFeatureMaps input feature maps dimension 4 [batch_size,channel,height,width]
    #outputFeatureMaps output feature maps
    #kernelsize default 3*3
    #add stride padding in future
    def __init__(self,lr_base = None,type_ = 'Dense',name ='',kernel_initializer = None,bias_initializer = None,activation = None):
        #updating w and b
        self.type = type_.upper()
        self.name = name
        self.phase = 'Test'
        self.copy_variable_to_optimizer = True
        self.a_ = 0.0
        norm_list = ['BATCHNORM','GROUPNORM']
        if activation is not None:
            self.activation = Activation(activation)
        if lr_base is not None:
            #to decide whether this layer needs kernel or not
            if (kernel_initializer is None or bias_initializer is None) and self.type not in norm_list:
                raise Exception('kernel_initializer/bias_initializer can not be None')
            if type(lr_base) == int:
                self.lr_base_weight = lr_base
                self.lr_base_bias = lr_base
            elif type(lr_base)==list and len(lr_base)==2:
                self.lr_base_weight = lr_base[0]
                self.lr_base_bias = lr_base[1]
            else:
                raise TypeError('learning rate mutiple must be int or tuple')
            self.variableNeedInitialize=True
            #print self.__variableNeedInitialize
        if kernel_initializer is not None:
            if type(kernel_initializer) == str:
                if kernel_initializer in ['Zeros','Ones','Constant','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling','LecunUniform','LecunNormal','GlorotNormal','GlorotUniform','HeNormal','HeUniform','Xavier']:
                    if kernel_initializer == 'Xavier':
                        kernel_initializer = 'GlorotNormal'
                    self.kernel_initializer = eval('initializers.'+kernel_initializer+'()')
                else:
                    raise TypeError("No such {} initializer defined, has to be one of them below\n 'Zeros','Ones','Constant','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling','LecunUniform','LecunNormal','GlorotNormal','GlorotUniform','HeNormal','HeUinform','Xavier'".format(kernel_initializer))
            elif issubclass(type(kernel_initializer),initializers.initializers):
                self.kernel_initializer = kernel_initializer
            else:
                raise Exception('kernel_initializer has to be defined correctly')
            self.w_ = np.array([])
        if bias_initializer is not None:
            if type(bias_initializer) == str:
                if bias_initializer in ['Zeros','Ones','Constant']:
                    self.bias_initializer = eval('initializers.'+bias_initializer+'()')
                else:
                    raise TypeError("{} is dismissed, has to be one of them below\n 'Zeros','Ones','Constant'".format(bias_initializer))
            elif type(bias_initializer) is initializers.Zeros or type(bias_initializer) is initializers.Ones or type(bias_initializer) is initializers.Constant:
                self.bias_initializer = bias_initializer
            else:
                raise Exception('bias_initializer has to be defined correctly')
            self.b_ = np.array([])
        
        
    def kernel_(self,kernelsizes):
        if type(kernelsizes) == list and len(kernelsizes) ==2:
            self.kernelheight = int(kernelsizes[0])
            self.kernelwidth = int(kernelsizes[1])
        elif type(kernelsizes) == int:
            self.kernelheight = kernelsizes
            self.kernelwidth = kernelsizes
        else:
            raise Exception('Kernelsize has to be int or tuple')
        
    def initiate_variable(self,*dims,**karg):
        if self.w_.shape[0] != 0:
            pass
        else:
            try:
                self.kernel_initializer.d1 = dims[0]
                self.kernel_initializer.d2 = dims[1]
                self.kernel_initializer.d3 = dims[2]
                self.kernel_initializer.d4 = dims[3]
            except:
                pass
            # fc
            if len(karg.keys()) != 0:
                self.bias_initializer.d2 = dims[1]
            else:
                if self.type == 'DEPTHWISECONV':
                    self.bias_initializer.d1 = dims[1]
                else:
                    self.bias_initializer.d1 = dims[0]
            if len(dims) == 2:
                n_in = dims[1]
                n_out = dims[0]
            elif len(dims) == 4:
                if self.type == 'DEPTHWISECONV':
                    n_in = dims[1]*dims[2]*dims[3]
                    n_out = dims[1]*dims[2]*dims[3]
                else:
                    n_in = dims[1]*dims[2]*dims[3]
                    n_out = dims[0]*dims[2]*dims[3]
            else:
                pass
                #raise Exception('Error of input dimension')

            try:
                self.kernel_initializer.n_in
                self.kernel_initializer.n_in = n_in
            except:
                pass
            try:
                self.kernel_initializer.n_out
                self.kernel_initializer.n_out = n_out
            except:
                pass
            w_ready = self.kernel_initializer.matrix_back()
            b_ready = self.bias_initializer.matrix_back()
            #fully connected
            if len(dims) == 2:
                w_ready = w_ready.squeeze(axis = (-1,-2))
                b_ready = b_ready.squeeze(axis = (-1,-2,-3))
            #batch_norm initializer
            if len(dims) == 1:
                w_ready = w_ready.squeeze()[np.newaxis,...]
                b_ready = b_ready.squeeze()[np.newaxis,...]
            self.w_ = w_ready.astype('float32')
            self.b_ = b_ready.astype('float32')
    #cooperation of tensors
   
    def __call__(self,X):
        #print X
        self.inFlow = X
        self.input = X.numpy()
        X.to_layer = self.name
        self.passForward()
        self.outFlow = Flow(self.a_,self.name)

        if self.phase.lower() == 'train':
            key_ = self.name
            self.optimizer.dataFlow.update({key_:X.from_layer})
            if self.type == 'BATCHNORM':
                self.optimizer.bn_param.update({self.name:self.bn_param})
            if self.copy_variable_to_optimizer:
                try:
                    self.optimizer.w_.update({self.name:self.w_})
                    self.optimizer.b_.update({self.name:self.b_})
                    
                except:
                    print("Layer {} doesn't have Learnable Varibles ".format(self.name))
                self.copy_variable_to_optimizer = False
        return self.outFlow
        
    def passForward(self):
        raise Exception('Layer {} has to define  passforward function'.format(self.type))
        pass
        
    def __update(self,name,delta):
        try:
            delta_base = self.optimizer.delta[name]
            delta += delta_base
        except:
            pass
        if delta is not None:
            self.optimizer.delta.update({name:delta})
        
    def back(self,delta,backLayerDelta):
        delta_return = self.backpropagate(delta,backLayerDelta)
        try:
            delta_back,delta_w_,delta_b_ = delta_return
            self.optimizer.delta_w_.update({self.name:delta_w_})
            self.optimizer.delta_b_.update({self.name:delta_b_})
        except:
            delta_back = delta_return
        if type(delta_back) != list:
            self.__update(self.name,delta_back)
        else:
            for i,lay_ in enumerate(delta_back):
                if self.type == 'ADD':
                    name_prev = self.mix_layers[i].name
                else:
                    name_prev = self.concat_layers[i].name
                
                self.__update(name_prev,lay_)
        
    def backpropagate(self,delta,backLayerDelta=True):
        raise Exception('Layer {} has to define backpropagate function'.format(self.type))
        pass
