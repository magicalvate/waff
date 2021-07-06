from optimizer import *
#learning_rate_eliminated
class RMSprop(Optimizer):
    def __init__(self,lr = 0.001,gamma = 0.9, epsilon = 1e-6):
        Optimizer.__init__(self,lr,'RMSprop')
        self.sqg_w_ = {}
        self.sqg_b_ = {}
        self.epsilon = epsilon
        self.gamma = gamma
    def calculate(self,key, profix = 'w'):
        profix +='_'
        try:
            sqg = eval('self.sqg_'+profix)[key]
        except:
            sqg = 0.0
        learning_rate = eval('self.lr_'+profix)[key]
        delta = eval('self.delta_'+profix)[key]
        sqg_t = self.gamma * sqg + (1-self.gamma)*delta*delta
        divid_g = np.sqrt(np.mean(sqg_t)+self.epsilon)
        delta_t = 1/divid_g*delta
        
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= delta_t*learning_rate
        eval('self.sqg_'+profix).update({key:sqg_t})
        
    def update_w(self):
        for key in self.w_.keys():
            self.calculate(key,'w')
           
            
    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
