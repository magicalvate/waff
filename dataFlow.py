import numpy as np
class Flow(object):
    def __init__(self,*arg):
        try:
            self.__value = arg[0]
            self.from_layer = arg[1]
            self.to_layer= arg[2]
        except:
            pass
    def numpy(self):
        return self.__value


