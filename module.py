# -*- coding: utf-8 -*-
#when tensor is not in straight flow
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


class Module(object):
    def __init__(self):
        pass
    def __call__(self,X):
        self.forward(X):
        self.__copy_variable_to_optimizer()
    def forward(self,X):
        raise Exception('Subclass {} has to define  passforward function'.format(self.type))
    
    def back(self,y):
        pass
    
