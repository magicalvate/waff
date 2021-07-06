from optimizer import *
class Adam(Optimizer):
    def __init__(self,lr = 0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-8):
        Optimizer.__init__(self,lr,'Adam')
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_w_ = {}
        self.m_b_ = {}
        self.v_w_ = {}
        self.v_b_ = {}
        self.t = 0
    def calculate(self,key, profix = 'w'):
        profix +='_'
        learning_rate = eval('self.lr_'+profix)[key]
        gt = eval('self.delta_'+profix)[key]
        
        try:
            mt = eval('self.m_'+profix)[key]
            vt = eval('self.v_'+profix)[key]
        except:
            mt = 0.0
            vt = 0.0
        mt = self.beta_1*mt +(1-self.beta_1)*gt
        vt = self.beta_2*vt +(1-self.beta_2)*gt*gt
        #print vt
        eval('self.m_'+profix).update({key:mt})
        eval('self.v_'+profix).update({key:vt})
        mt_1 = mt/(1-self.beta_1**self.t)
        vt_1= vt/(1-self.beta_2**self.t)
        #coeff = mt_1/(np.sqrt(vt_1+self.epsilon))
        coeff = mt_1/(np.sqrt(vt_1)+self.epsilon)
        #print coeff
        eval('self.'+profix)[key] =(1-self.weight_decay)*eval('self.'+profix)[key]
        eval('self.'+profix)[key] -= coeff*learning_rate
        
    def update(self):
        self.t +=1
        self.update_w()
        self.update_b()
        
    def update_w(self):
        for key in self.w_.keys():
            self.calculate(key,'w')
    def update_b(self):
        for key in self.b_.keys():
            self.calculate(key,'b')
