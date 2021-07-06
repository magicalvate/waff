# -*- coding: utf-8 -*-
def create_layername(name,type_ ,namelist):
    #print type_.lower()
    if type_.lower() == 'dense':
        type_ = 'fc'
        #print type_
    n = len(type_)
    
    if type(name) == str and len(name)>0:
        if name.lower() not in namelist and name[:n].lower()==type_.lower():
            return name.lower()
        else:
            compare_list = [name_[:len(type_)+1] for name_ in namelist]
            i = 1
            while type_.lower()+str(i) in compare_list:
                i += 1
            name = type_.lower()+str(i)
            return name
    else:
        compare_list = [name_[:len(type_)+1] for name_ in namelist]
        i = 1
        while type_.lower()+str(i) in compare_list:
            i += 1
        name = type_.lower()+str(i)
        return name





    
