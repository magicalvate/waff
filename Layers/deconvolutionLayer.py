#coding:utf-8
#author:yifan ding
import numpy as np
from waff.Layers.ops.mat_calucation import Conv2d
from waff.Layers.convolutionLayer import Conv
class Deconv(Conv):
    #inputFeatureMaps input feature maps dimension 4 [batch_size,channel,height,width]
    #outputFeatureMaps output feature maps
    #kernelsize default 3*3
    
    #add stride padding in future 
    def __init__(self,kernelnumber,kernelsizes,padding='valid',strides=1,lr_base=[1,2]\
        ,destrides=1,name = '',kernel_initializer ='GlorotNormal' ,bias_initializer ='Zeros',activation = None):
        Conv.__init__(self,kernelnumber=kernelnumber,kernelsizes=kernelsizes\
            ,padding=padding,strides=strides,lr_base=lr_base,name = name,kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,activation = activation )
        self.type = 'DECONV'
        if destrides>=1:
            self.destridesType='fractional'
        else:
            self.destridesType='deconv'
        self.destrides=destrides
        
    def __fractional_padd(self,delta):
        zero_=self.strides-1
        last_pad_col=(self.padded_inputFeatureMaps.shape[-1]-self.kernelwidth)%self.strides
        last_pad_row=(self.padded_inputFeatureMaps.shape[-2]-self.kernelheight)%self.strides
        batch_size,channel,height,width=delta.shape
        height=(height-1)*zero_+height+last_pad_row
        width=(width-1)*zero_+width+last_pad_col
        zeros_=np.zeros((batch_size,channel,height,width))
        #for j in range(delta.shape[2]):
        #    for i in range(delta.shape[3]):
        #        zeros_[:,:,j*self.strides,i*self.strides]=delta[:,:,j,i]
        zeros_[:,:,::self.strides,::self.strides]=delta
        return zeros_

    def __forward_fractional_padd(self):
        #the convolutional type is fractional 
        if self.destridesType=='fractional':
            zero_=self.destrides
            height=(self.height-1)*zero_+self.height
            width=(self.width-1)*zero_+self.width
            zeros_=np.zeros((self.batch_size,self.inputChannelSize,height,width))
            #for j in range(self.height):
            #    for i in range(self.width):
            #        zeros_[:,:,j*(self.destrides+1),i*(self.destrides+1)]=self.input[:,:,j,i]
            zeros_[:,:,::(self.destrides+1),::(self.destrides+1)]=self.input
        else:
            zeros_=self.input.copy()

        return np.pad(zeros_,pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')
    
    def __initiateVariables(self):
        self.initiate_variable(self.kernelnumber,self.inputChannelSize,self.kernelheight,self.kernelwidth)
        self.variableNeedInitialize=False
        
    def __conv2d(self,x,w,b = None,strides = 1, mode = 'forward'):
        return Conv2d(x,w,b,strides,mode)
    
    def passForward(self):
        (self.batch_size,self.inputChannelSize,self.height,self.width)=self.input.shape
        if self.variableNeedInitialize:
            #print 'this initiateStep Processed'
            #print self.variableNeedInitialize
            self.__initiateVariables()

        self.padded_inputFeatureMaps=self.__forward_fractional_padd()
        self.a_=np.array(self.__conv2d(self.padded_inputFeatureMaps,self.w_,self.b_,self.strides))
    
        try:
            self.a_ = self.activation(self.a_)
        except:
            pass
    
    def __calculateDeltaBackLayer(self,delta):
        (channelThisLayer,channelBackLayer,filterH,filterW)=np.array(self.w_).shape
        paddingH=self.kernelheight-1
        paddingW=self.kernelwidth-1
        if self.strides!=1:
            #print 'strides  is not equal to 1***************'
            fraction_delta = self.__fractional_padd(delta)
            
        else:
            fraction_delta= delta
        delta_padded=np.pad(fraction_delta,pad_width=((0,0),(0,0),(paddingH,paddingH),(paddingW,paddingW)),mode='constant')
       
        w_rot180 = self.__rotate180(self.w_)
        w_rot180 = w_rot180.swapaxes(0,1)
        
        deltaBackLayer = self.__conv2d(delta_padded,w_rot180,mode='delta_back')
        if self.padding!=0:
            deltaBackLayer = deltaBackLayer[:,:,self.padding:-self.padding,self.padding:-self.padding]
        if self.destridesType == 'fractional':
            deltaBackLayer = deltaBackLayer[:,:,::(self.destrides+1),::(self.destrides+1)]
        return deltaBackLayer
    def __rotate180(self,w):
        w1 = w.reshape(-1,np.product(w.shape[2:]))
        w1 = w1[:,::-1]
        return w1.reshape(w.shape)
        
    def backpropagate(self,delta,backLayerDelta=True):
        try:
            delta = self.activation.backpropagate(delta)
        except:
            pass
        #delta for BackLayer should be calculated before w updated
        if backLayerDelta:
            #print 'this step excuted'
            deltaBackLayer = self.__calculateDeltaBackLayer(delta)
        else:
            deltaBackLayer = None
        if self.strides != 1:
            delta = self.__fractional_padd(delta)
        delta_w_ = self.__conv2d(self.padded_inputFeatureMaps,delta,mode = 'delta_w')
        assert delta_w_.shape == self.w_.shape
        delta_b_= np.sum(delta,axis=(0,2,3)).reshape(-1,1,1,1)
        delta_w_ /= self.batch_size
        delta_b_ /= self.batch_size
        assert delta_b_.shape == self.b_.shape
        return deltaBackLayer,delta_w_,delta_b_





