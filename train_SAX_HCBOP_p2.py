import pandas as pd
import numpy as np
import warnings
import time
import argparse

from data.data_loader import CT_data_loader, ai4i2020_data_loader
from classifier.SAX_HCBOP import SAX_HCBOP_parameters
from classifier.SAX_HCBOP.SAX_HCBOP_p2 import SAX_HCBOP_p2
warnings.filterwarnings('ignore')
    
### Parameters

parser = argparse.ArgumentParser()
#### dataset settings
parser.add_argument('--dataset', type=str, default="CT", help='dataset name, CT or ai4i2020')
## long TS dataset settings
parser.add_argument('--len_TS', type=int, help='length for an instance, unit:day.')
parser.add_argument('--ywindow', type=int, help='length for the y window, unit:day.')
parser.add_argument('--resample', type=int, default=6, help='resample time, unit:hour.')
## model settings:
parser.add_argument('--model', type=str, default="SAX_HCBOP", help='Model name')
parser.add_argument('--symbol', type=str, default="A,B,C,D", help='Number of attention heads.')
## training settings
parser.add_argument('--sp_ratio', type=int, default=35, help='The ratio of splitting training and test sets')
parser.add_argument('--num_para_p2', type=int, default=5, help='The number of random search parameter combinations (KRR)')
parser.add_argument('--num_para_KRR', type=int, default=100000, help='The number of random search parameter combinations for KRR')
args = parser.parse_args()

args.symbol = [sym for sym in args.symbol.split(",")]

if args.dataset == "CT":
    Parameter_KRR = SAX_HCBOP_parameters.parameter_generation_LightGBM(args.resample, args.num_para_KRR, args.dataset) 
    Name = ['Oil','Voltage','Current','Energy']
    rawdata = CT_data_loader(str(args.resample)+'H')
elif args.dataset == "ai4i2020":
    Parameter_KRR = SAX_HCBOP_parameters.parameter_generation_LightGBM(args.resample, args.num_para_KRR, args.dataset) 
    Name = ['AT', 'PT', 'RS', 'T', 'TW']
    rawdata = ai4i2020_data_loader()    

save_path = "result_A/" + "SAX" + "_" + args.dataset + "_" + str(args.sp_ratio) + "_"


find_para = np.load(save_path + 'parameters_p1.npy')
if args.model == "SAX_HCBOP":
    result = np.load(save_path + 'result_1_p1.npy')
elif args.model == "SAX_HBOP":
    result = np.load(save_path + 'result_2_p1.npy')
elif args.model == "SAX_CBOP":
    result = np.load(save_path + 'result_3_p1.npy')
    
start = time.time()

result_tuple_list = []
for num_cyc in range(args.num_para_p2):
    result_tuple = SAX_HCBOP_p2(model = args.model, 
                                rawdata = rawdata, 
                                Name = Name,
                                sp_ratio = args.sp_ratio/100,
                                symbol = args.symbol,
                                Parameter_KRR = Parameter_KRR,
                                find_para = find_para,
                                result = result
                                )
    Parameter_KRR = result_tuple[-1]
    find_para = np.concatenate([find_para, np.array(result_tuple[0]).reshape(1, -1)], axis=0)
    result = np.concatenate([result, np.array(result_tuple[2])], axis=0)
    result_tuple_list.append(result_tuple)
     
end = time.time()

find_para_p2, ins_sta_p2, result_p2, FI_LA_p2, CutPoint_LA_p2, pred_p2 = [], [], [], [], [], []
for num_cyc in range(args.num_para_p2):
    find_para_p2.append(result_tuple_list[num_cyc][0])
    ins_sta_p2.append(result_tuple_list[num_cyc][1])
    result_p2.append(result_tuple_list[num_cyc][2])
    FI_LA_p2.append(result_tuple_list[num_cyc][3])
    CutPoint_LA_p2.append(result_tuple_list[num_cyc][4])
    pred_p2.append(result_tuple_list[num_cyc][5])

para_p2 = np.vstack(find_para_p2)
ins_sta_p2 = np.concatenate(ins_sta_p2, axis=0)
result_p2 = np.concatenate(result_p2, axis=0)

save_path = "result_A/" + args.model + "_" + args.dataset + "_" + str(args.sp_ratio) + "_"
np.save(save_path+'parameters_p2.npy', para_p2)
np.save(save_path+'instance_sta_p2.npy', ins_sta_p2)
np.save(save_path+'result_p2.npy', result_p2)


# For Result_1
Result_1 = pd.DataFrame()
for i in range(args.num_para_p2):
    for ii in range(1):
        Re = CutPoint_LA_p2[i][ii]
        Re['Location'] = "%d%d.pickle"%(i,ii)
        Result_1 = pd.concat([Result_1, Re])
Result_1.to_csv(save_path+'cutpoint_p2.csv', index=False)

# For Result_2
Result_2 = pd.DataFrame()
for i in range(args.num_para_p2):
    for ii in range(1):
        Re = FI_LA_p2[i][ii]
        
        c_name_1 = 'feature' + str(i) + str(ii)
        c_name_2 = 'gain' + str(i) + str(ii)
        c_name_3 = 'split' + str(i) + str(ii)
        
        Re.columns = [c_name_1, c_name_2, c_name_3]
        Result_2 = pd.concat([Result_2, Re], axis = 1)
Result_2.to_csv(save_path+'FI_p2.csv', index=False)

# For Result_3
Result_3 = pd.DataFrame([[start, end, end-start]], columns = ['Start', 'End', 'Time'])
Result_3.to_csv(save_path+'time_p2.csv', index=False)

# For Result_4
Result_4 = np.stack(pred_p2, axis=0)     
np.save(save_path+'pred_result_p2.npy', Result_4)


