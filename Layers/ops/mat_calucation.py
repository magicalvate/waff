import numpy as np
import time

def split_arr(input_data, filter_h, filter_w, stride=1):
    N, C, H, W = input_data.shape
    out_h = (H  - filter_h) // stride + 1
    out_w = (W  - filter_w) // stride + 1

    shapes  = (N,C,out_h,out_w,filter_h,filter_w)
    strides = list(input_data.strides+input_data.strides[-2:])
    strides[2] *= stride
    strides[3] *= stride
    xx = np.lib.stride_tricks.as_strided(input_data, shape=shapes, strides=strides)
    return xx
def Pool2dMaxBack(delta,x,maxbase,k_h,k_w,stride = 1):
    delta_back = np.zeros_like(x)
    N,C,out_h,out_w = delta.shape
    r_ = maxbase.reshape(-1)
    c_ = np.arange(N*C*out_h*out_w)
    zeros = np.zeros((N*C*out_h*out_w,k_h*k_w)).astype(x.dtype)
    zeros[c_,r_]=1
    zeros = zeros.reshape(N,C,out_h,out_w,k_h,k_w)
    for i in range(k_h):
        imax = i + stride*out_h
        for j in range(k_w):
            jmax = j + stride*out_w
            delta_back[:,:,i:imax:stride,j:jmax:stride]+= delta*zeros[:,:,:,:,i,j]
    return delta_back
def Pool2dAverageBack(delta,x,k_h,k_w,stride = 1):
    delta_back = np.zeros_like(x)
    N,C,out_h,out_w = delta.shape
    for i in range(k_h):
        imax = i + stride*out_h
        for j in range(k_w):
            jmax = j + stride*out_w
            delta_back[:,:,i:imax:stride,j:jmax:stride]+= delta/(k_h*k_w)
    return delta_back
    
def Pool2dAverage(x,k_h,k_w,stride):
    z = split_arr(x,k_h,k_w,stride)
    z = np.sum(z,axis=(4,5))/(k_h*k_w)
    return z
    
def Pool2dMax(x,k_h,k_w,stride,phase = 'test'):
    z = split_arr(x,k_h,k_w,stride)
    if phase.lower() == 'train':
        max_base = z.reshape(-1,k_w*k_h)
        max_base = np.argmax(max_base,axis = -1)
        z = np.max(z,axis=(4,5))
        return z,max_base.reshape(z.shape)
    else:
        z = np.max(z,axis=(4,5))
        return z

def Matmul(w,a,b = 0,phase = 0):
    try:
        t = b.shape[0]
    except:
        t = 0
    if t != 0:
        z = np.dot(a,w.transpose(1,0))+b
    else:
        if w.shape[0] == a.shape[0] and phase == 0:
            w = w.transpose(1,0)
            z = np.dot(w,a)
        else:
            z = np.dot(a,w)
    return z
    
def _im2col(input_data, filter_h, filter_w, stride=1):
    N, C, H, W = input_data.shape
    out_h = (H  - filter_h) // stride + 1
    out_w = (W  - filter_w) // stride + 1

    shapes  = (N,C,out_h,out_w,filter_h,filter_w)
    strides = list(input_data.strides+input_data.strides[-2:])
    strides[2] *= stride
    strides[3] *= stride
    xx = np.lib.stride_tricks.as_strided(input_data, shape=shapes, strides=strides)
    return xx
    
    
    
def Conv2ddepthwise(x,w,b = None,strides = 1, mode = 'forward'):
    if mode =='forward' or mode == 'delta_back':
        N, C ,H, W  = x.shape
        C_out,C_in,filter_h,filter_w = w.shape
        x = _im2col(x,filter_h,filter_w,strides)
        x = x.transpose(0,2,3,1,4,5)
        z = np.sum(x*w,axis = (-1,-2)).transpose(0,3,1,2)
    elif mode == 'delta_w':
        N, C_in, H_in, W_in = x.shape
        N, C_out, H_out, W_out = w.shape
        x = _im2col(x,H_out,W_out,strides)
        x = x.transpose(0,2,3,1,4,5)
        z =  np.sum(x*w[:,np.newaxis,np.newaxis,...],axis = (0,-1,-2),keepdims = True).squeeze((-1,-2)).transpose(0,3,1,2)
    else:
        raise Exception('No mode defined')
    if b is not None:
        b = b.squeeze(-1)
        z += b
    return z

def Conv2d(x,w,b = None,strides = 1, mode = 'forward'):
    if mode =='forward' or mode == 'delta_back':
        N, C ,H, W  = x.shape
        C_out,C_in,filter_h,filter_w = w.shape
        x = _im2col(x,filter_h,filter_w,strides)
        z = np.tensordot(x, w, [(1, 4, 5), (1, 2, 3)]).transpose(0, 3, 1, 2)

    elif mode == 'delta_w':
        N, C_in, H_in, W_in = x.shape
        N, C_out, H_out, W_out = w.shape
        x = _im2col(x,H_out,W_out,strides)
        z = np.tensordot(x, w, [(0, 4, 5), (0, 2, 3)]).transpose(3, 0, 1, 2)
    else:
        raise Exception('No mode defined')
    if b is not None:
        b = b.squeeze(-1)
        z += b
    return z

