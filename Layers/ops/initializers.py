import numpy as np

#method for initializer 'Zeros','Ones','Constant','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling','lecun_uniform','lecun_normal','glorot_normal','glorot_uniform','he_normal'
class initializers(object):
    def __init__(self,method = 'normal'):
        self.d1 = 1
        self.d2 = 1
        self.d3 = 1
        self.d4 = 1
        self.method = method
    def matrix_back(self):
        pass
class Zeros(initializers):
    def __init__(self):
        initializers.__init__(self,'zeros')
    
    def matrix_back(self):
        return np.zeros((self.d1,self.d2,self.d3,self.d4))
class Ones(initializers):
    def __init__(self):
        initializers.__init__(self,'ones')
    def matrix_back(self):
        return np.ones((self.d1,self.d2,self.d3,self.d4))
        
class Constant(initializers):
    def __init__(self,value = 1.0):
        initializers.__init__(self,'constant')
        self.value = value
        
    def matrix_back(self):
        return np.ones((self.d1,self.d2,self.d3,self.d4))*self.value
    
class RandomNormal(initializers):
    def __init__(self,mean = 0.0,stddev = 0.05, seed = None):
        initializers.__init__(self,'random_normal')
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
    def matrix_back(self):
        np.random.seed(self.seed)
        return (np.random.randn(self.d1,self.d2,self.d3,self.d4)-self.mean)*self.stddev

class RandomUniform(initializers):
    def __init__(self,minval = -0.05,maxval = 0.05, seed = None):
        initializers.__init__(self,'randomn_uniform')
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
    def matrix_back(self):
        np.random.seed(self.seed)
        return np.random.uniform(self.minval,self.maxval,(self.d1,self.d2,self.d3,self.d4))

class TruncatedNormal(initializers):
    def __init__(self,mean = 0.0, stddev = 0.05, seed = None):
        initializers.__init__(self,'truncated_normal')
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
    def matrix_back(self):
        def trunc(length):
            np.random.seed(self.seed)
            matrix = np.random.randn(length)
            unsatisfy = len(np.where(np.abs(matrix)>2)[0])
            if unsatisfy > 0:
                sub = trunc(unsatisfy)
                matrix[np.where(np.abs(matrix)>2)]=sub
                return matrix
            else:
                return matrix
        length = self.d1*self.d2*self.d3*self.d4
        #print length
        matrix = trunc(length)
        matrix = matrix.reshape(self.d1,self.d2,self.d3,self.d4)
        matrix = (matrix - self.mean)*self.stddev
        return matrix
        
class VarianceScaling(initializers):
    def __init__(self,scale = 1.0,mode = 'fan_in',distribution = 'normal',seed = None):
        initializers.__init__(self,'variance_scaling')
        if mode not in ['fan_in','fan_out','fan_avg']:
            raise TypeError("mode must be 'fan_in','fan_out' or 'fan_avg'")
        else:
            self.mode = mode
        if distribution not in ['normal','uniform']:
            raise TypeError("mode must be 'normal' or 'uniform'")
        else:
            self.distribution = distribution
        self.seed = seed
        self.scale = scale
        self.n_in = 0.0
        self.n_out = 0.0
    
    def matrix_back(self):
        def trunc(length):
            np.random.seed(self.seed)
            matrix = np.random.randn(length)
            unsatisfy = len(np.where(np.abs(matrix)>2)[0])
            if unsatisfy > 0:
                sub = trunc(unsatisfy)
                matrix[np.where(np.abs(matrix)>2)]=sub
                return matrix
            else:
                return matrix
        if self.mode == 'fan_in':
            n = self.n_in
        elif self.mode == 'fan_out':
            n = self.n_out
        else:
            n = np.mean([self.n_in,self.n_out])
        if self.distribution == 'normal':
            length = self.d1*self.d2*self.d3*self.d4
            matrix = trunc(length)
            matrix *= np.sqrt(self.scale*1.0/n)
            matrix = matrix.reshape(self.d1,self.d2,self.d3,self.d4)
        else:
            limit = np.sqrt(3.0*self.scale/n)
            np.random.seed(self.seed)
            matrix = np.random.uniform(-limit,limit,(self.d1,self.d2,self.d3,self.d4))
        #print matrix.shape
        return matrix

class LecunUniform(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'lecun_uniform')
        self.seed = seed
        self.n_in = 0.0
        
    def matrix_back(self):
        n = self.n_in
        limit = np.sqrt(3.0/n)
        np.random.seed(self.seed)
        return np.random.uniform(-limit,limit,(self.d1,self.d2,self.d3,self.d4))
        
class LecunNormal(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'lecun_normal')
        self.seed = seed
        self.n_in = 0.0
    def matrix_back(self):
        n = self.n_in
        np.random.seed(self.seed)
        return np.random.randn(self.d1,self.d2,self.d3,self.d4)*np.sqrt(1.0/n)
        
# xavier initializer
class GlorotNormal(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'glorot_normal')
        self.seed = seed
        self.n_in = 0.0
        self.n_out = 0.0
    def matrix_back(self):
        n = self.n_in + self.n_out
        np.random.seed(self.seed)
        return np.random.randn(self.d1,self.d2,self.d3,self.d4)*np.sqrt(2.0/n)

class GlorotUniform(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'glorot_uniform')
        self.seed = seed
        self.n_in = 0.0
        self.n_out = 0.0
    def matrix_back(self):
        n = self.n_in + self.n_out
        limit = np.sqrt(6.0/n)
        np.random.seed(self.seed)
        return np.random.uniform(-limit,limit,(self.d1,self.d2,self.d3,self.d4))

class HeNormal(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'he_normal')
        self.seed = seed
        self.n_in = 0.0
    def matrix_back(self):
        n = self.n_in
        np.random.seed(self.seed)
        return np.random.randn(self.d1,self.d2,self.d3,self.d4)*np.sqrt(2.0/n)

class HeUniform(initializers):
    def __init__(self,seed = None):
        initializers.__init__(self,'he_uniform')
        self.seed = seed
        self.n_in = 0.0
    
    def matrix_back(self):
        n = self.n_in
        limit = np.sqrt(6.0/n)
        np.random.seed(self.seed)
        return np.random.uniform(-limit,limit,(self.d1,self.d2,self.d3,self.d4))
