import pandas as pd
import numpy as np
import random
   
def Offset(data, para_list_o, Name):
    data_mean = pd.DataFrame()
    for i in range(len(Name)):
        column = data[[Name[i]]]
        column.columns = ['item']
        column_mean = column.item.rolling(para_list_o[i], min_periods=1).mean()
        column_mean = pd.DataFrame(column_mean)
        data_mean = pd.concat([data_mean, column_mean], axis=1)
    data_mean.columns = Name
    result = data-data_mean
    return result

def Thres(data, para_list_t, Name):
    result = pd.DataFrame()
    for i in range(len(Name)):
        thr = para_list_t[i]
        column = data[Name[i]]
        column['max_sta'] = column.apply(lambda x : x-thr if x > thr else 0)
        result = pd.concat([result, column['max_sta']], axis=1)
    return result

def y_time(target_np, ywindow):
    y_all =  np.zeros(len(target_np), dtype = np.int32)
    for i in range(target_np.shape[0]):
        if target_np[i] == 1:
            if i-ywindow+1 >= 0:
                y_all[i-ywindow+1:i+1]=1
            else:
                y_all[:i+1]=1
    return y_all

def XY(data):
    Y = data[['Label']]
    X = data.drop('Label', axis = 1)
    Y.index = range(len(Y))
    X.index = range(len(X))
    Y.drop([0],inplace = True)
    X.drop([len(X)-1],inplace = True)
    Y.index = range(len(Y))
    X.index = range(len(X))
    Result = pd.concat([X,Y],axis=1)
    return Result

def KFold_ordered(COPY_y, K_fold, num_machine_sta, w_x, cp = None):
    
    if cp is None:
        cp = [i/(K_fold+1) for i in range(1, K_fold+1)]
    else:
        cp = cp
        K_fold = len(cp)
        
    train_index_list, val_index_list = [], []
    class_ins_sta = np.zeros([1, K_fold, 6], dtype = np.int32)
    for age in range(K_fold):
        start = 0
        train_index_l, val_index_l = [], []
        for machine in range(len(num_machine_sta)):
            num_ins = num_machine_sta[machine]
            index = [i for i in range(start, start+num_ins)]
            train_index = index[w_x-1 : int(num_ins*cp[age])]
            val_index = index[int(num_ins*cp[age]) :]
            
            train_index_l += train_index
            val_index_l += val_index
            start += num_ins
            
        ind_0, ind_1 = [], []
        for ind in val_index_l:
            ind_0.append(ind) if COPY_y[ind] == 0 else ind_1.append(ind)
        class_ins_sta[0, age, 0] = len(ind_0)
        class_ins_sta[0, age, 1] = len(ind_1)
        
        ind_0, ind_1 = [], []
        for ind in train_index_l:
            ind_0.append(ind) if COPY_y[ind] == 0 else ind_1.append(ind)
        class_ins_sta[0, age, 2] = len(ind_0)
        class_ins_sta[0, age, 3] = len(ind_1)
        
        ind_1 = ind_1 + random.choices(ind_1, k=len(ind_0)-len(ind_1))
        class_ins_sta[0, age, 4] = len(ind_0)
        class_ins_sta[0, age, 5] = len(ind_1)
        
        train_index_l = ind_0+ind_1
        
        train_index_list.append(train_index_l), val_index_list.append(val_index_l)
    return train_index_list, val_index_list, class_ins_sta
