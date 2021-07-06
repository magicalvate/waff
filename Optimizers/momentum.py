from optimizer import Optimizer
class Momentum(Optimizer):
    def __init__(self, lr = 0.01,momentum = 0.9):
        Optimizer.__init__(self, lr,'Momentum')
        self.momentum_w_= {}
        self.momentum_b_ = {}
        self.br = momentum
    
    
    def calculate(self,key, profix = 'w'):
        profix +='_'
        learning_rate = eval('self.lr_'+profix)[key]
        delta = eval('self.delta_'+profix)[key]
        try:
            moment = eval('self.momentum_'+profix)[key]
        except:
            moment = 0.0
        moment = moment*self.br + delta
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= moment*learning_rate
        eval('self.momentum_'+profix).update({key:moment})
    def update_w(self):
    #print 'this step'
        for key in self.w_.keys():
            self.calculate(key,'w')
            '''
            learning_rate = self.lr_w[key]
            delta_w_ = self.delta_w_[key]
            try:
                moment = self.momentum[key]
                #print 'moment exists'
            except:
                moment = 0.0
                #print 'create'
                
            moment = moment*self.br + learning_rate*delta_w_
            self.w_[key] = (1-self.weight_decay)*self.w_[key]
            self.w_[key] -= moment
            self.momentum.update({key:moment})
            '''
    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
           
