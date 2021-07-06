import numpy as np
from waff.Layers.layer import Layer

class BatchNorm(Layer):
    def __init__(self,name = '',beta = 0.0,gamma =1.0,epsilon = 1e-5,moment = 0.99,affine = True):
        Layer.__init__(self,1, 'BATCHNORM', name,'Ones','Ones')
        #learnable variables
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.moment = moment
        self.bn_param = {}
        self.affine = affine
        #self.__variableInitialize = True
    def __initiateVariables(self,num,c):
        if num == 4:
            self.initiate_variable(1,c,1,1,type_ = 'batch_norm')
        else:
            self.initiate_variable(c)
           
        self.w_ *= self.gamma
        self.b_ *= self.beta
        self.variableNeedInitialize=False
    def passForward(self):
        x = self.input
        if self.variableNeedInitialize:
            self.__initiateVariables(len(x.shape),x.shape[1])
        batch_size = x.shape[0]
        
        
        running_mean = self.bn_param.get('mean',0.0)
        running_var = self.bn_param.get('var',1.0)

        if self.phase.lower() == 'train':
            if len(x.shape) == 4:
                axis = (0,-1,-2)
            else:
                axis = 0
            x_mean = np.mean(x,axis = axis,keepdims = True)
            x_var = np.var(x,axis =  axis,keepdims = True)
            running_mean = self.moment*running_mean + (1-self.moment)*x_mean
            #print running_mean.shape
            running_var = self.moment*running_var + (1-self.moment)*x_var
            
            self.derivate =1/np.sqrt(x_var+self.epsilon)
            self.x_normalized = (x-x_mean)*self.derivate
            #self.derivate =1/np.sqrt(running_var+self.epsilon)
            #self.x_normalized = (x-running_mean)*self.derivate
            self.bn_param.update({'mean':running_mean})
            self.bn_param.update({'var':running_var})
            self.x_mean = x_mean
            self.x_var = x_var

        #self.phase is test
        else:
            self.x_normalized = (x-running_mean)/np.sqrt(running_var+self.epsilon)

        self.a_ = self.x_normalized*self.w_ + self.b_
        #self.a_ = self.x_normalized*gamma + beta
        
    def backpropagate(self,delta,backLayerDelta = True):
        batch_size = delta.shape[0]
        if len(delta.shape) == 4:
            N = delta.shape[0]*delta.shape[-1]*delta.shape[-2]
            axis = (0,-1,-2)
        else:
            N = batch_size
            axis = 0
            
        delta_w_ = np.sum(delta*self.x_normalized,axis=axis,keepdims=True)
        delta_b_ = np.sum(delta,axis=axis,keepdims = True)
    
        
        if backLayerDelta:
            dx_norm = delta*self.w_
            dvar = -0.5*np.sum(dx_norm*(self.input-self.x_mean),axis = axis,keepdims = True)*np.power(self.x_var+self.epsilon,-1.5)
            dmean = -np.sum(dx_norm*self.derivate,axis = axis,keepdims = True)
            dmean += dvar*np.sum(-2*(self.input-self.x_mean),axis = axis,keepdims = True)/N
            deltaBackLayer = dx_norm*self.derivate+dvar*2*(self.input-self.x_mean)/N + dmean/N
        else:
            deltaBackLayer = None
        
        assert delta_w_.shape == self.w_.shape
        assert delta_b_.shape == self.b_.shape
        delta_w_ /= batch_size
        delta_b_ /= batch_size
        if self.affine:
            return deltaBackLayer,delta_w_,delta_b_
        else:
            return deltaBackLayer

class GroupNorm(Layer):
    def __init__(self,num_group,name = '',beta = 0.0,gamma =1.0,epsilon = 1e-5,moment = 0.99,affine = True):
        Layer.__init__(self,1,'GROUPNORM',name,'Ones','Ones')
        #learnable variables
        self.gamma = gamma
        self.beta = beta
        self.g = num_group
        self.epsilon = epsilon
        self.moment = moment
        self.affine = affine
    def __initiateVariables(self,c):
    
        self.initiate_variable(1,c,1,1,type_ = 'group_norm')
        self.w_ *= self.gamma
        self.b_ *= self.beta
        self.variableNeedInitialize=False
        
    def passForward(self):
        x = self.input
        assert len(self.input.shape)==4
        N,C,H,W  = x.shape
        if self.variableNeedInitialize:
            self.__initiateVariables(x.shape[1])
        assert C%self.g == 0
        x_group = np.reshape(x,(N,self.g,C//self.g,H,W))
        #print x_group.shape
        x_mean = np.mean(x_group,axis = (2,3,4),keepdims = True)
        x_var = np.var(x_group,axis =(2,3,4),keepdims = True)
     
        
        self.derivate =1/np.sqrt(x_var+self.epsilon)
        self.x_normalized = ((x_group-x_mean)*self.derivate).reshape(N,C,H,W)
        
        self.x_mean = x_mean
        self.x_var = x_var


        self.a_ = self.x_normalized*self.w_ + self.b_
        #self.a_ = self.x_normalized*gamma + beta
        
    def backpropagate(self,delta,backLayerDelta = True):
        N,C,H,W = delta.shape
        assert len(delta.shape) == 4
        axis = (0,2,3)
        N_g = C//self.g*H*W
        delta_w_ = np.sum(delta*self.x_normalized,axis=axis,keepdims=True)
        delta_b_ = np.sum(delta,axis=axis,keepdims = True)
    
        
        if backLayerDelta:
            dx_norm = delta*self.w_
            dx_groupnorm = dx_norm.reshape(N,self.g,C//self.g,H,W)
            x_group = self.input.reshape((N, self.g, C //self.g, H, W))
            dvar = -0.5*np.sum(dx_groupnorm*(x_group-self.x_mean),axis = (2,3,4),keepdims = True)*np.power(self.x_var+self.epsilon,-1.5)
            dmean = -np.sum(dx_groupnorm*self.derivate,axis = (2,3,4),keepdims = True)
            dmean += dvar*np.sum(-2*(x_group-self.x_mean),axis = (2,3,4),keepdims = True)/N_g
            deltaBackLayer = dx_groupnorm*self.derivate+dvar*2*(x_group-self.x_mean)/N_g + dmean/N_g
            deltaBackLayer = deltaBackLayer.reshape((N, C, H, W))
        else:
            deltaBackLayer = None
        #print delta_w_.shape,delta_b_.shape
        assert delta_w_.shape == self.w_.shape
        assert delta_b_.shape == self.b_.shape
        #if self.affine:
        #else:
        delta_w_ /= N
        delta_b_ /= N
        if self.affine:
            return deltaBackLayer,delta_w_,delta_b_
        else:
            return deltaBackLayer
