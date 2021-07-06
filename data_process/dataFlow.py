import numpy as np
class Flow(object):
    def __init__(self,input,info = None,bottom = None):
        self.__value = input
        if info is not None:
            self.info = info
        if bottom is not None:
            self.bottom = bottom

    def numpy(self):
        return self.__value
    def update(self,x):
        assert type(x) == np.ndarray
        self.__value = x


        
class flow_info(object):
    def __init__(self,data_flow,phase):
        self.pair_info =[]
        self.phase = phase
