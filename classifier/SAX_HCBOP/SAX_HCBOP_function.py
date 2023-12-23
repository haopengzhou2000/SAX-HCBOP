import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge

def generate_words(length, alphabet, prefix='', words=[]):
    if len(prefix) == length:
        words.append(prefix)
    else:
        for char in alphabet:
            new_prefix = prefix + char
            generate_words(length, alphabet, new_prefix, words)
    return words

def SplitPoint_EWidth(data_x, data_y, Tra_ind, Name, Symbol_size, num_his):
    X_Train = pd.DataFrame(data_x[Tra_ind])
    Y_Train = pd.DataFrame(data_y[Tra_ind], columns = ['Label'])
    Col_Y = Y_Train.copy()
    Col_Y.index = range(len(Col_Y))
    Class = [i for i in range(num_his)]
    Result = pd.DataFrame()

    for FD in range(len(Name)-len(Name)//4):
        Col_X = X_Train[FD]
        
        Max = Col_X.max()
        Min = Col_X.min()
        d = (Max-Min)/num_his
        Candidate_Cutpoints = [-np.inf] + [Min+i*d for i in range(1,num_his)] + [np.inf]
        
        Cut = pd.cut(Col_X, bins = Candidate_Cutpoints, labels = Class)
        Col_X = pd.DataFrame(Cut, columns = [FD])
        Cut = pd.concat([Col_X, Col_Y], axis = 1)
        Cut.columns = ['F','Label']
        
        SP1 = SP_find(Cut, Class, Symbol_size)
        SP1 = [Candidate_Cutpoints[sp] for sp in SP1]
        SP1 = pd.DataFrame(SP1, columns = [[FD]])
        Result = pd.concat([Result,SP1], axis=1)

    for FD in range(len(Name)-len(Name)//4,len(Name)):
        Col_X = X_Train[[FD]]
        Col_A = pd.concat([Col_X, Col_Y], axis = 1)
        Col_A.columns = ['F','Label']
        Col_B = Col_A[Col_A.F!= 0] 
        Col_X = Col_B['F']
        Col_Y = Col_B[['Label']]
        
        Max = Col_X.max()
        Min = Col_X.min()
        d = (Max-Min)/num_his
        Candidate_Cutpoints = [-np.inf] + [Min+i*d for i in range(1,num_his)] + [np.inf]
        
        Cut = pd.cut(Col_X, bins = Candidate_Cutpoints, labels = Class)
        Col_X = pd.DataFrame(Cut, columns = ['F'])
        Cut = pd.concat([Col_X, Col_Y], axis = 1)
        
        SP1 = SP_find(Cut, Class, Symbol_size-1)
        SP1 = [Candidate_Cutpoints[sp] for sp in SP1]
        SP1 = [float(Min/2)] + SP1
        SP1 = pd.DataFrame(SP1, columns = [[FD]])
        Result = pd.concat([Result,SP1], axis=1)
    return Result

def SplitPoint_ENumber(data_x, data_y, Tra_ind, Name, Symbol_size, num_his):
    X_Train = pd.DataFrame(data_x[Tra_ind])
    Y_Train = pd.DataFrame(data_y[Tra_ind], columns = ['Label'])
    Col_Y = Y_Train.copy()
    Col_Y.index = range(len(Col_Y))
    Result = pd.DataFrame()

    for FD in range(len(Name)-len(Name)//4):
        Class = []
        Candidate_Cutpoints = []
        Col_X = X_Train[[FD]]
        Col_X2 = Col_X.sort_values(FD)
        Col_X2.index = range(0,len(Col_X2))
        for a in range(1, num_his):
            SP1 = Col_X2.loc[round(a*len(Col_X2)/num_his)][FD]
            SP2 = Col_X2.loc[round(a*len(Col_X2)/num_his)+1][FD]
            SP_F = (SP1 + SP2) / 2 + a*0.00001
            Candidate_Cutpoints.append(SP_F)
        
        Candidate_Cutpoints = [-np.inf] + sorted(list(set(Candidate_Cutpoints))) + [np.inf]
        Class = [i for i in range(len(Candidate_Cutpoints)-1)]
        Col_X = Col_X[FD]
        
        Cut = pd.cut(Col_X, bins = Candidate_Cutpoints, labels = Class)
        Col_X = pd.DataFrame(Cut, columns = [FD])
        Cut = pd.concat([Col_X, Col_Y], axis = 1)
        Cut.columns = ['F','Label']
        
        SP1 = SP_find(Cut, Class, Symbol_size)
        SP1 = [Candidate_Cutpoints[sp] for sp in SP1]
        SP1 = pd.DataFrame(SP1, columns = [[FD]])
        Result = pd.concat([Result,SP1], axis=1)
     
        
    for FD in range(len(Name)-len(Name)//4,len(Name)):
        Class = []
        Col_X = X_Train[[FD]]
        Col_A = pd.concat([Col_X, Col_Y], axis = 1)
        Col_A.columns = ['F','Label']
        Col_B = Col_A[Col_A.F!= 0] 
        Col_X = Col_B[['F']]
        Min = Col_X.min()
        Candidate_Cutpoints = []
        
        Col_X2 = Col_X.sort_values('F')
        Col_X2.index = range(0,len(Col_X2))
        for a in range(1, num_his):
            SP1 = Col_X2.loc[round(a*len(Col_X2)/num_his)]['F']
            SP2 = Col_X2.loc[round(a*len(Col_X2)/num_his)+1]['F']
            SP_F = (SP1 + SP2) / 2 + a*0.00001
            Candidate_Cutpoints.append(SP_F)
            
        Candidate_Cutpoints = [-np.inf] + sorted(list(set(Candidate_Cutpoints))) + [np.inf]
        Class = [i for i in range(len(Candidate_Cutpoints)-1)]
        Col_X = Col_X['F']
        
        Cut = pd.cut(Col_X, bins = Candidate_Cutpoints, labels = Class)
        Col_X = pd.DataFrame(Cut, columns = ['F'])
        Cut = pd.concat([Col_X, Col_Y], axis = 1)
        
        SP1 = SP_find(Cut, Class, Symbol_size-1)
        SP1 = [Candidate_Cutpoints[sp] for sp in SP1]
        SP1 = [float(Min/2)] + SP1
        SP1 = pd.DataFrame(SP1, columns = [[FD]])
        Result = pd.concat([Result,SP1], axis=1)
    return Result

class Node():
    def __init__(self, his_index):
        self.his_index = his_index 
        self.IG = -np.inf
        self.sp = -1
        
def Ent(D, C0, C1):
    if C0 == 0 or C1 ==0:
        Result = 0
    else:
        Result = -(C0/D)*(math.log2(C0/D)) - (C1/D)*(math.log2(C1/D))
    return Result

def IGS(NF, current_node):
    Index_L = current_node.his_index[0]
    Index_U = current_node.his_index[-1]
    
    if Index_U == Index_L:
        sp = -2
        IG = -2
    else:
        NF_node = NF[Index_L:Index_U+1]
        D = np.sum(np.array(NF_node)) 
        C0 = np.sum(np.array(NF_node['N0']))  
        C1 = D-C0                            
        Ent_Before= Ent(D, C0, C1)       
        
        sp = (Index_L + Index_U + 2)//2
        IG = 0
        CL0 = 0
        CL1 = 0
        for i in range(Index_L+1, Index_U+1):
            CL0 = CL0 + NF.loc[i-1]['N0']
            CL1 = CL1 + NF.loc[i-1]['N1']
            DL = CL0 + CL1
            CR0 = C0-CL0
            CR1 = C1-CL1
            DR = CR0 + CR1
            Ent_After = ((DL/D)*Ent(DL, CL0, CL1) + 
                         (DR/D)*Ent(DR, CR0, CR1))
            IG_N = Ent_Before - Ent_After
            if IG_N > IG:
                sp = i         
                IG = IG_N
                
    current_node.sp = sp
    current_node.IG = IG
    return current_node

def SP_find(Data, Class, Symbol_size):
    NA, NB, Data_G = [], [], []
    for b in range(0,len(Class)): 
        C1 = Data[Data['F'].isin([b])]  
        C1_0 = C1[C1['Label'].isin([0])]
        C1_1 = C1[C1['Label'].isin([1])]
        N0 = len(C1_0)
        N1 = len(C1_1)
        NA.append(N0), NB.append(N1), Data_G.append(C1)
    NA = pd.DataFrame(NA, columns = ['N0'])
    NB = pd.DataFrame(NB, columns = ['N1'])
    NF = pd.concat([NA,NB], axis=1)
    
    
    best_split_point = []
    Node_root = Node(his_index = Class)
    
    Node_root = IGS(NF, Node_root)
    best_split_point.append(Node_root.sp)
    
    while len(best_split_point) < Symbol_size-1:
        Node_list = []
        for i in range(len(best_split_point)+1):
            if i == 0:
                Node_list.append(Node(his_index = Node_root.his_index[min(Node_root.his_index):best_split_point[i]]))
            elif i == len(best_split_point):
                Node_list.append(Node(his_index = Node_root.his_index[best_split_point[i-1]:max(Node_root.his_index)+1]))
            else:
                Node_list.append(Node(his_index = Node_root.his_index[best_split_point[i-1]:best_split_point[i]]))
        
        maximum_IG = -10
        for node in Node_list:
            node = IGS(NF, node)
            if node.IG > maximum_IG:
                best_split_point_candi = node.sp
                maximum_IG = node.IG
        best_split_point.append(best_split_point_candi)
        best_split_point.sort()
    return best_split_point

def SplitPoint2_EWidth(data_x, data_y, Tra_ind, Name, Symbol_size):
    X_Train = pd.DataFrame(data_x[Tra_ind])
    Y_Train = pd.DataFrame(data_y[Tra_ind], columns = ['Label'])
    Col_Y = Y_Train.copy()
    Col_Y.index = range(len(Col_Y))
    Result = pd.DataFrame()
    
    for FD in range(len(Name)-len(Name)//4):
        Col_X = X_Train[FD]
        Max = Col_X.max()
        Min = Col_X.min()
        d = (Max-Min)/Symbol_size
        Candidate_Cutpoints = [Min+i*d for i in range(1,Symbol_size)]
        SP1 = pd.DataFrame(Candidate_Cutpoints, columns = ['Num'])
        Result = pd.concat([Result,SP1], axis=1)
    
    for FD in range(len(Name)-len(Name)//4,len(Name)):
        Col_X = X_Train[[FD]]
        Col_A = pd.concat([Col_X, Col_Y], axis = 1)
        Col_A.columns = ['F','Label']
        Col_B = Col_A[Col_A.F!= 0] 
        Col_X = Col_B['F']
        Col_Y = Col_B[['Label']]
        
        Max = Col_X.max()
        Min = Col_X.min()
        d = (Max-Min)/(Symbol_size-1)
        Candidate_Cutpoints = [float(Min/2)] + [Min+i*d for i in range(1,Symbol_size-1)]
        SP1 = pd.DataFrame(Candidate_Cutpoints, columns = ['Num'])
        Result = pd.concat([Result,SP1], axis=1)
    return Result


def SplitPoint2_ENumber(data_x, data_y, Tra_ind, Name, Symbol_size):
    X_Train = pd.DataFrame(data_x[Tra_ind])
    Y_Train = pd.DataFrame(data_y[Tra_ind], columns = ['Label'])
    Col_Y = Y_Train.copy()
    Col_Y.index = range(len(Col_Y))
    Result = pd.DataFrame()
    
    for FD in range(0,len(Name)-4):
        SP = []
        Col_X = X_Train[[FD]]
        Col_X = Col_X.sort_values(FD)
        Col_X.index = range(0,len(Col_X))
        for a in range(1, Symbol_size):
            SP1 = Col_X.loc[round(a*len(Col_X)/Symbol_size)][FD]
            SP2 = Col_X.loc[round(a*len(Col_X)/Symbol_size)+1][FD]
            SP_F = (SP1 + SP2) / 2 + a*0.00001
            SP.append(SP_F)
        SP1 = pd.DataFrame(SP, columns = ['Num'])
        SP1.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
        Result = pd.concat([Result,SP1], axis=1)
    
    for FD in range(len(Name)-4,len(Name)):
        Col_X = X_Train[[FD]]
        Col_A = pd.concat([Col_X, Col_Y], axis = 1)
        Col_A.columns = ['F','Label']
        Col_B = Col_A[Col_A.F!= 0] 
        Col_X = Col_B['F']
        Min = Col_X.min()
        SP = [float(Min/2)]
        
        Col_X = Col_B[['F']]
        Col_X = Col_X.sort_values('F')
        Col_X.index = range(0,len(Col_X))
        for a in range(1, Symbol_size-1):
            SP1 = Col_X.loc[round(a*len(Col_X)/Symbol_size-1)]['F']
            SP2 = Col_X.loc[round(a*len(Col_X)/Symbol_size-1)+1]['F']
            SP_F = (SP1 + SP2) / 2 + a*0.00001
            SP.append(SP_F)
        SP1 = pd.DataFrame(SP, columns = ['Num'])
        SP1.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
        Result = pd.concat([Result,SP1], axis=1)
    return Result

def BagofWord_f(data, Para, CP, Name_all, Class, Ind):
    len_word = int(Para[len(Name_all)//4*6+1])
    Word_L= generate_words(len_word, Class, prefix='', words=[])
    Bag_of_Word_para_list, Bag_of_Word_para_temp_list = [], []
    for p in range(len(Name_all)): 
        Word = pd.DataFrame(np.zeros((len(Ind), len(Word_L))), columns = [Word_L])
        Word_temp = pd.DataFrame(np.zeros((len(Ind), len(Word_L))), columns = [Word_L])
        
        data_F = data[:, p]
        w = int(Para[len(Name_all)//4*2+p]) 
        Coeff = [1+(i+1)/(w-len_word+2) for i in range(w-len_word+1)]
        cutpoints = [-np.inf] + [CP.loc[i][p] for i in range(len(Class)-1)] + [np.inf]
        class_trans = np.array(pd.cut(data_F, bins = cutpoints, labels = Class))
        for t in range(len(Ind)):
            Index = Ind[t]
            for W_n in range(w-len_word+1):
                s_word = [class_trans[Index - w + W_n + num_w] for num_w in range(1, len_word+1)]
                W = "".join(s_word)
                Word.loc[t,W] = Word.loc[t][W] + 1
                Word_temp.loc[t,W] = Word_temp.loc[t][W] + Coeff[W_n]
                
        Bag_of_Word_para_list.append(np.array(Word)), Bag_of_Word_para_temp_list.append(np.array(Word_temp))
    Bag_of_Word_para = np.concatenate(Bag_of_Word_para_list, axis = 1)
    Bag_of_Word_para_temp = np.concatenate(Bag_of_Word_para_temp_list, axis = 1)
    return Bag_of_Word_para_temp, Bag_of_Word_para

def BagofWord(data, Para, CP, Name_all, Class, Ind):
    len_word = int(Para[len(Name_all)//4*6+1])
    Word_L= generate_words(len_word, Class, prefix='', words=[])
    Bag_of_Word_para_list = []
    for p in range(len(Name_all)): 
        Word = pd.DataFrame(np.zeros((len(Ind), len(Word_L))), columns = [Word_L])
        
        data_F = data[:, p]
        w = int(Para[len(Name_all)//4*2+p]) 
        cutpoints = [-np.inf] + [CP.loc[i][p] for i in range(len(Class)-1)] + [np.inf]
        class_trans = np.array(pd.cut(data_F, bins = cutpoints, labels = Class))
        for t in range(len(Ind)):
            Index = Ind[t]
            for W_n in range(w-len_word+1):
                s_word = [class_trans[Index - w + W_n + num_w] for num_w in range(1, len_word+1)]
                W = "".join(s_word)
                Word.loc[t,W] = Word.loc[t][W] + 1
        Bag_of_Word_para_list.append(np.array(Word))
    Bag_of_Word_para = np.concatenate(Bag_of_Word_para_list, axis = 1)
    return Bag_of_Word_para

def BagofWord_coeff(data, Para, CP, Name_all, Class, Ind):
    len_word = int(Para[len(Name_all)//4*6+1])
    Word_L= generate_words(len_word, Class, prefix='', words=[])
    Bag_of_Word_para_temp_list = []
    for p in range(len(Name_all)): 
        Word_temp = pd.DataFrame(np.zeros((len(Ind), len(Word_L))), columns = [Word_L])
        
        data_F = data[:, p]
        w = int(Para[len(Name_all)//4*2+p])  
        Coeff = [1+(i+1)/(w-len_word+2) for i in range(w-len_word+1)]
        cutpoints = [-np.inf] + [CP.loc[i][p] for i in range(len(Class)-1)] + [np.inf]
        class_trans = np.array(pd.cut(data_F, bins = cutpoints, labels = Class))
        for t in range(len(Ind)):
            Index = Ind[t]
            for W_n in range(w-len_word+1):
                s_word = [class_trans[Index - w + W_n + num_w] for num_w in range(1, len_word+1)]
                W = "".join(s_word)
                Word_temp.loc[t,W] = Word_temp.loc[t][W] + Coeff[W_n]
                
        Bag_of_Word_para_temp_list.append(np.array(Word_temp))
    Bag_of_Word_para_temp = np.concatenate(Bag_of_Word_para_temp_list, axis = 1)
    return Bag_of_Word_para_temp

def ParaPre(X_Para, y_Para, Target):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_Para)
    gp = KernelRidge(kernel="poly", degree=1)
    gp.fit(scaler.transform(X_Para), y_Para)
    
    preds = gp.predict(scaler.transform(Target))
    max_index = np.argmax(preds)
    return max_index





