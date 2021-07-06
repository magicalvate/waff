#coding:utf-8
#author:yifan ding
import numpy as np
from waff.Layers.ops.mat_calucation import Conv2ddepthwise
from waff.Layers.layer import Layer
import time
class DepthwiseConv(Layer):
    #inputFeatureMaps input feature maps dimension 4 [batch_size,channel,height,width]
    #outputFeatureMaps output feature maps
    #kernelsize default 3*3
    def __init__(self,kernelsizes,padding='valid',strides=1,lr_base=[1,2],name = '',kernel_initializer ='GlorotNormal' ,bias_initializer ='Zeros',activation = None):
        Layer.__init__(self,lr_base,'DEPTHWISECONV',name,kernel_initializer,bias_initializer,activation)
        self.strides=strides
        self.kernel_(kernelsizes)
        if type(padding) == str:
            if padding.lower()=='valid':
                self.padding=0
            elif padding.lower()=='same':
                self.padding= (self.kernelwidth-1)/2
            else:
                raise TypeError("'padding' must be integral or 'same' or 'valid'")
        elif type(padding) == int:
            self.padding=padding
        else:
            raise TypeError("'padding' must be integral or 'same' or 'valid'")
        
    def passForward(self):
        (self.batch_size,self.inputChannelSize,self.height,self.width)=self.input.shape
        if self.variableNeedInitialize:
            self.initiate_variable(1,self.inputChannelSize,self.kernelheight,self.kernelwidth)
            self.variableNeedInitialize=False
            
        if self.padding != 0:
            padded_map =np.pad(self.input,pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')
            
        else:
            padded_map = self.input
   
        self.a_= Conv2ddepthwise(padded_map,self.w_,self.b_,self.strides)

        try:
            self.a_ = self.activation(self.a_)
        except:
            pass


    def __rotate180(self,w):
        w1 = w.reshape(-1,np.product(w.shape[2:]))
        w1 = w1[:,::-1]
        return w1.reshape(w.shape)


    def __fractional_padd(self,delta):
        zero_=self.strides-1
        last_pad_col=(self.width-self.kernelwidth+2*self.padding)%self.strides
        last_pad_row=(self.height-self.kernelheight+2*self.padding)%self.strides
        batch_size,channel,height,width=delta.shape
        height=(height-1)*zero_+height+last_pad_row
        width=(width-1)*zero_+width+last_pad_col
        zeros_=np.zeros((batch_size,channel,height,width)).astype(self.input.dtype)
        zeros_[:,:,::self.strides,::self.strides]=delta
        return zeros_
    
    def __calculateDeltaBackLayer(self,delta):
        #(channelThisLayer,channelBackLayer,filterH,filterW)=np.array(self.w_).shape
        paddingH=self.kernelheight-1
        paddingW=self.kernelwidth-1
        delta_padded=np.pad(delta,pad_width=((0,0),(0,0),(paddingH,paddingH),(paddingW,paddingW)),mode='constant')
        w_rot180 = self.__rotate180(self.w_)
        #w_rot180 = w_rot180.swapaxes(0,1)
        deltaBackLayer = Conv2ddepthwise(delta_padded,w_rot180,mode='delta_back')
        if self.padding!=0:
            deltaBackLayer = deltaBackLayer[:,:,self.padding:-self.padding,self.padding:-self.padding]
        assert deltaBackLayer.shape == self.input.shape
        return deltaBackLayer
        
    def backpropagate(self,delta,backLayerDelta=True):
        assert delta.shape == self.a_.shape
        try:
            delta = self.activation.backpropagate(delta)
        except:
            pass
        if self.strides != 1:
            delta = self.__fractional_padd(delta)
        if backLayerDelta:
            deltaBackLayer = self.__calculateDeltaBackLayer(delta)
        else:
            deltaBackLayer = None
        if self.padding!=0:
            padded_inputFeatureMaps=np.pad(self.input,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')
        else:
            padded_inputFeatureMaps=self.input
        
        delta_w_ = Conv2ddepthwise(padded_inputFeatureMaps,delta,mode = 'delta_w')
        #print delta_w_.dtype
        assert delta_w_.shape == self.w_.shape
        delta_b_= np.sum(delta,axis=(0,2,3)).reshape(-1,1,1,1)
        assert delta_b_.shape == self.b_.shape
        #%********************** can result diff
        delta_w_ /= self.batch_size
        delta_b_ /= self.batch_size
        return deltaBackLayer, delta_w_,delta_b_
