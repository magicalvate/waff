# -*- coding: utf-8 -*-
import pickle
#from waff.net import Net
def save_model(net,save_path = 'model.pkl'):
    pickle_file = open(save_path,'wb')
    pickle.dump(net,pickle_file)
    pickle_file.close()
    
def load_model(load_path):
    pickle_file = open(load_path,'rb')
    net_from_load = pickle.load(pickle_file)
    pickle_file.close()
    return net_from_load
    



    
