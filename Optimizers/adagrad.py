from optimizer import *
#automatic decreasing learning_rate
#benefit of this optimizer manully adjustion of learning_rate is eliminate
class AdaGrad(Optimizer):
    def __init__(self,lr = 0.01,epsilon = 1e-8):
        Optimizer.__init__(self, lr,'AdaGrad')
        self.r_w_ = {}
        self.r_b_ = {}
        self.epsilon = epsilon
    def calculate(self,key, profix = 'w'):
        profix +='_'
        learning_rate = eval('self.lr_'+profix)[key]
        delta = eval('self.delta_'+profix)[key]
        try:
            r = eval('self.r_'+profix)[key]
        except:
            r = 0.0
        r += delta*delta
        sqrt_r = np.sqrt(r+self.epsilon)
        delta /= sqrt_r
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= delta*learning_rate
        eval('self.r_'+profix).update({key:r})
    def update_w(self):
    #print 'this step'
        for key in self.w_.keys():
            self.calculate(key,'w')
           
            
    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
