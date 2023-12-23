import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from classifier.SAX_HCBOP import data_function
from classifier.SAX_HCBOP import SAX_HCBOP_function

def SAX_HCBOP_p1(model,
                 rawdata,
                 Name,
                 sp_ratio,
                 symbol,
                 ParaList
                 ):
    
    Name_d = [name+"_d" for name in Name]
    Name_o = [name+"_o" for name in Name]
    Name_t = [name+"_t" for name in Name]
    Name_all = Name + Name_d + Name_o + Name_t
    
    num_machine = len(rawdata['Machine'].value_counts())
    COPY_x, COPY_y = [], []
    for num in range(num_machine): 
        x = rawdata[rawdata['Machine'].isin([num+1])]
        x_d = x[Name].diff().bfill()
        x_d.columns = Name_d
        x_o = data_function.Offset(x[Name], [int(ParaList[p_id]) for p_id in range(0,len(Name))], Name)
        x_o.columns = Name_o
        x_t = data_function.Thres(x[Name], [ParaList[p_id] for p_id in range(len(Name),2*len(Name))], Name)
        x_t.columns = Name_t
        
        x_data = pd.concat([x[Name], x_d, x_o, x_t], axis=1)
        x['Label'] = data_function.y_time(x['Label'].values, int(ParaList[len(Name)*6+2]))
        x = pd.concat([x_data, x[['Label', 'Machine']]], axis=1)
        #x = data_function.XY(x)
        
        COPY_x.append(x[Name_all+['Machine']].values), COPY_y.append(x[['Label']].values)
    COPY_x = np.vstack(COPY_x)
    COPY_y = np.vstack(COPY_y)
    
    _, num_machine_sta = np.unique(COPY_x[:, -1], return_counts=True)
    
    train_index_list, val_index_list, class_ins_sta = data_function.KFold_ordered(COPY_y, 1, num_machine_sta, int(max(ParaList[len(Name)*2 : len(Name)*6])), [sp_ratio])
        
    evaluation = np.zeros([1, 1, 4], dtype = np.float64)
    
    train_index = train_index_list[0]
    val_index = val_index_list[0]
            
    data = COPY_x[:, :-1]
    label = COPY_y
            
    if model == "SAX_CBOP":
        if int(ParaList[len(Name)*6]) == 0:
            Cut_Point = SAX_HCBOP_function.SplitPoint2_EWidth(data, label, train_index, Name_all, len(symbol))
        elif int(ParaList[len(Name)*6]) == 1:
            Cut_Point = SAX_HCBOP_function.SplitPoint2_ENumber(data, label, train_index, Name_all, len(symbol))
    else:
        if int(ParaList[len(Name)*6]) == 0:
            Cut_Point = SAX_HCBOP_function.SplitPoint_EWidth(data, label, train_index, Name_all, len(symbol), 10)
        elif int(ParaList[len(Name)*6]) == 1:
            Cut_Point = SAX_HCBOP_function.SplitPoint_ENumber(data, label, train_index, Name_all, len(symbol), 10)
    
    if model == "SAX_HBOP":
        X_train = SAX_HCBOP_function.BagofWord(data, ParaList, Cut_Point, Name_all, symbol, train_index)
    else:
        X_train = SAX_HCBOP_function.BagofWord_coeff(data, ParaList, Cut_Point, Name_all, symbol, train_index)
    y_train = label[train_index]
    
    lgb_Tra = lgb.Dataset(X_train, y_train)
    
    boost_round = 50
    params = {
        'boosting': 'gbdt', 
        'objective': 'binary', 
        'max_depth': int(ParaList[-3]),
        'num_leaves': int(ParaList[-2]), 
        'learning_rate': ParaList[-1], 
        'feature_fraction': 0.8, 
        'bagging_fraction': 0.8,
        'bagging_freq': 1, 
        'verbose': 0, 
        }
    gbm = lgb.train(params, lgb_Tra, num_boost_round=boost_round)
    
    if model == "SAX_HBOP":
        X_test = SAX_HCBOP_function.BagofWord(data, ParaList, Cut_Point, Name_all, symbol, val_index)
    else:
        X_test = SAX_HCBOP_function.BagofWord_coeff(data, ParaList, Cut_Point, Name_all, symbol, val_index)
    y_test = label[val_index]
    
    y_pred = (gbm.predict(X_test) >= 0.5).astype(int)
    
    evaluation[0, 0, 0] = accuracy_score(y_test, y_pred)
    evaluation[0, 0, 1] = recall_score(y_test, y_pred)
    evaluation[0, 0, 2] = precision_score(y_test, y_pred)
    evaluation[0, 0, 3] = f1_score(y_test, y_pred)
    
    Word_L= SAX_HCBOP_function.generate_words(int(ParaList[len(Name)*6+1]), symbol, prefix='', words=[])
    FI = pd.DataFrame({'feature': [name+word for name in Name_all for word in Word_L], 'gain': gbm.feature_importance('gain'), 'split': gbm.feature_importance('split')})
    
    return ParaList, class_ins_sta, evaluation, [FI], [Cut_Point], np.stack([y_test.squeeze(), y_pred], axis=1)

def SAX_HCBOP_p2(model, 
                 rawdata,
                 Name,
                 sp_ratio,
                 symbol,
                 Parameter_KRR,
                 find_para,
                 result
                 ):
    
    next_para_ind = SAX_HCBOP_function.ParaPre(find_para, np.mean(result, axis = 1)[:, -1], Parameter_KRR)
    ParaList = Parameter_KRR[next_para_ind]
    Parameter_KRR = np.delete(Parameter_KRR, next_para_ind, axis=0)
    
    ParaList, class_ins_sta, evaluation, FI, CutPoint, pred = SAX_HCBOP_p1(model = model,
                                                                           rawdata = rawdata, 
                                                                           Name = Name,
                                                                           sp_ratio = sp_ratio,
                                                                           symbol = symbol,
                                                                           ParaList = ParaList
                                                                           )
    
    return ParaList, class_ins_sta, evaluation, FI, CutPoint, pred, Parameter_KRR