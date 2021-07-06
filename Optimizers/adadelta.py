from optimizer import *
#learning_rate_eliminated
class Adadelta(Optimizer):
    def __init__(self,gamma = 0.9, epsilon = 1e-6):
        Optimizer.__init__(self,lr = None, type_= 'Adadelta')
        self.sqg_w_ = {}
        self.sqg_b_ = {}
        self.sqx_w_ = {}
        self.sqx_b_ = {}
        self.epsilon = epsilon
        self.gamma = gamma
    def calculate(self,key, profix = 'w'):
        profix +='_'
        
        try:
            sqg = eval('self.sqg_'+profix)[key]
            sqx = eval('self.sqx_'+profix)[key]
        except:
            sqg = 0.0
            sqx = 0.0
        delta = eval('self.delta_'+profix)[key]
        sqg_t = self.gamma * sqg + (1-self.gamma)*delta*delta
        divid_g = np.sqrt(np.mean(sqg_t)+self.epsilon)
        divid_x = np.sqrt(np.mean(sqx)+self.epsilon)
        delta_t = divid_x/divid_g*delta
        sqx_t = self.gamma * sqx + (1-self.gamma)*delta_t*delta_t
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= delta_t
        eval('self.sqg_'+profix).update({key:sqg_t})
        eval('self.sqx_'+profix).update({key:sqx_t})

    def update_w(self):
    #print 'this step'
        for key in self.w_.keys():
            self.calculate(key,'w')
           
            
    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
