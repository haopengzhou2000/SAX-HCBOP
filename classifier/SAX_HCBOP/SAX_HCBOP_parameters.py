import numpy as np
import random

def CT_parameter_generation(resample):
    w_p = 24/resample
    date_list = [7, 10, 13] #unit:day
    y_win_list = [3, 5, 7] #unit:day
    date_list = [time*w_p for time in date_list]
    y_win_list = [(time-1)*w_p+1 for time in y_win_list]
    
    Para = []
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    
    Para.append(random.choice([320, 323, 326]))
    Para.append(random.choice([120000, 135000, 150000]))
    Para.append(random.choice([0.17, 0.20, 0.23]))
    Para.append(random.choice([70000000, 90000000, 110000000]))
    
    
    O = random.choice(date_list)
    V = random.choice(date_list)
    C = random.choice(date_list)
    P = random.choice(date_list)
    
    for _ in range (4):
        Para.append(O)
        Para.append(V)
        Para.append(C)
        Para.append(P)
    
    Para.append(random.choice([0,1]))
    Para.append(random.choice([2,3]))
    Para.append(random.choice(y_win_list))
    return Para

def ai4i2020_parameter_generation():
    date_list = [5, 7, 9] 
    y_win_list = [1]

    Para = []
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    Para.append(random.choice(date_list))
    
    Para.append(random.choice([300.1, 301.5])) # 50% and 75%
    Para.append(random.choice([310.1, 311.1]))
    Para.append(random.choice([1503, 1612]))
    Para.append(random.choice([40.1, 46.8]))
    Para.append(random.choice([108, 162]))
    
    p1 = random.choice(date_list)
    p2 = random.choice(date_list)
    p3 = random.choice(date_list)
    p4 = random.choice(date_list)
    p5 = random.choice(date_list)
    
    for _ in range (4):
        Para.append(p1)
        Para.append(p2)
        Para.append(p3)
        Para.append(p4)
        Para.append(p5)

    Para.append(random.choice([0,1]))
    Para.append(random.choice([2,3]))
    Para.append(random.choice(y_win_list))
    return Para

def parameter_generation_LightGBM(resample, num_para, dataset):
    result = []
    for _ in range(num_para):
        if dataset == "CT":
            Para = CT_parameter_generation(resample)
        elif dataset == "ai4i2020":
            Para = ai4i2020_parameter_generation()
            
        Para.append(random.choice([3,5,10]))
        Para.append(random.choice([5,15,25]))
        Para.append(random.choice([0.01, 0.05, 0.1]))
                
        result.append(np.array(Para))
    result = np.vstack(result)
    return result
