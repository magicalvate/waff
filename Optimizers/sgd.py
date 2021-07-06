from optimizer import Optimizer
class SGD(Optimizer):
    def __init__(self,lr = 0.01):
        Optimizer.__init__(self, lr,'SGD')
    def update_w(self):
        assert len(self.w_.keys()) == len(self.delta_w_.keys())
        for key in self.w_.keys():
            self.w_[key] = (1-self.weight_decay)*self.w_[key]
            learning_rate = self.lr_w_[key]
            delta_w_ = self.delta_w_[key]
            self.w_[key] -= learning_rate*delta_w_
    def update_b(self):
        assert len(self.b_.keys()) == len(self.delta_b_.keys())
        for key in self.b_.keys():
            self.b_[key] =(1-self.weight_decay)*self.b_[key]
            learning_rate = self.lr_b_[key]
            delta_b_ = self.delta_b_[key]
            self.b_[key] -= learning_rate*delta_b_
