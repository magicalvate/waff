from optimizer import Optimizer
class Nesterov(Optimizer):
    def __init__(self,lr = 0.01, momentum = 0.9,beta = 0.9):
        Optimizer.__init__(self, lr,'Nesterov')
        self.momentum_w_= {}
        self.momentum_b_ = {}
        self.br = momentum
        self.beta = beta
        self.delta_previous_w_ = {}
        self.delta_previous_b_ = {}
    
    def calculate(self,key, profix = 'w'):
        profix +='_'
        learning_rate = eval('self.lr_'+profix)[key]
        delta = eval('self.delta_'+profix)[key]
        try:
            moment = eval('self.momentum_'+profix)[key]
            delta_t = eval('self.delta_previous_'+profix)[key]
        except:
            moment = 0.0
            delta_t = 0.0
        moment = moment*self.br + delta +self.beta*(delta-delta_t)
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= moment*learning_rate
        eval('self.momentum_'+profix).update({key:moment})
        eval('self.delta_previous_'+profix).update({key:delta})
    def update_w(self):
    #print 'this step'
        for key in self.w_.keys():
            self.calculate(key,'w')

    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
           
