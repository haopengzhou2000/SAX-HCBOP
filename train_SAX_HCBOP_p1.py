import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
import time
import argparse

from data.data_loader import CT_data_loader, ai4i2020_data_loader
from classifier.SAX_HCBOP import SAX_HCBOP_parameters
from classifier.SAX_HCBOP.SAX_HCBOP_p1 import SAX_HCBOP_p1
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    ### Parameters
    parser = argparse.ArgumentParser()
    #### dataset settings
    parser.add_argument('--dataset', type=str, default="CT", help='dataset name, CT or ai4i2020')
    ## long TS dataset settings
    parser.add_argument('--len_TS', type=int, help='length for an instance, unit:hour.')
    parser.add_argument('--ywindow', type=int, help='length for the y window, unit:hour.')
    parser.add_argument('--resample', type=int, default=6, help='resample time, unit:hour.')
    ## model settings:
    parser.add_argument('--symbol', type=str, default="A,B,C,D", help='symbol (string)')
    ## training settings
    parser.add_argument('--sp_ratio', type=int, default=35, help='The ratio of splitting training and test sets')
    parser.add_argument('--max_workers', type=int, default=5, help='How many cpus are used')
    parser.add_argument('--num_para_p1', type=int, default=10, help='The number of random search parameter combinations')
    args = parser.parse_args()
    
    args.symbol = [sym for sym in args.symbol.split(",")]
    
    if args.dataset == "CT":
        Parameter = SAX_HCBOP_parameters.parameter_generation_LightGBM(args.resample, args.num_para_p1, args.dataset) 
        Name = ['Oil','Voltage','Current','Energy']
        rawdata = CT_data_loader(str(args.resample)+'H')  # dataframe column: [feature, label, machine]
    elif args.dataset == "ai4i2020":
        Parameter = SAX_HCBOP_parameters.parameter_generation_LightGBM(args.resample, args.num_para_p1, args.dataset) 
        Name = ['AT', 'PT', 'RS', 'T', 'TW']
        rawdata = ai4i2020_data_loader()    
    
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=args.max_workers)
    
    result_tuple_list = []
    for num_cyc in range(args.num_para_p1):
        result_tuple = pool.submit(SAX_HCBOP_p1, rawdata, Name, args.sp_ratio/100, args.symbol, Parameter[num_cyc])
        result_tuple_list.append(result_tuple)
    
    pool.shutdown()
    end = time.time()
    
    
    find_para, ins_sta, result_1, result_2, result_3, FI_LA, CutPoint_LA, pred = [], [], [], [], [], [], [], []
    for num_cyc in range(args.num_para_p1):
        find_para.append(result_tuple_list[num_cyc].result()[0])
        ins_sta.append(result_tuple_list[num_cyc].result()[1])
        result_1.append(result_tuple_list[num_cyc].result()[2])
        result_2.append(result_tuple_list[num_cyc].result()[3])
        result_3.append(result_tuple_list[num_cyc].result()[4])
        FI_LA.append(result_tuple_list[num_cyc].result()[5])
        CutPoint_LA.append(result_tuple_list[num_cyc].result()[6])
        pred.append(result_tuple_list[num_cyc].result()[7])
    
    para_p1 = np.vstack(find_para)
    ins_sta_p1 = np.concatenate(ins_sta, axis=0)
    result_1_p1 = np.concatenate(result_1, axis=0)
    result_2_p1 = np.concatenate(result_2, axis=0)
    result_3_p1 = np.concatenate(result_3, axis=0)
    
    save_path = "result_A/" + "SAX" + "_" + args.dataset + "_" + str(args.sp_ratio) + "_"
    np.save(save_path+'parameters_p1.npy', para_p1)
    np.save(save_path+'instance_sta_p1.npy', ins_sta_p1)
    np.save(save_path+'result_1_p1.npy', result_1_p1)
    np.save(save_path+'result_2_p1.npy', result_2_p1)
    np.save(save_path+'result_3_p1.npy', result_3_p1)
    
    
    # For Result_1
    Result_1 = pd.DataFrame()
    for i in range(args.num_para_p1):
        for ii in range(1):
            Re = CutPoint_LA[i][ii]
            Re['Location'] = "%d%d.pickle"%(i,ii)
            Result_1 = pd.concat([Result_1, Re])
    Result_1.to_csv(save_path+'cutpoint_p1.csv', index=False)
    
    # For Result_2
    Result_2 = pd.DataFrame()
    for i in range(args.num_para_p1):
        for ii in range(1):
            Re = FI_LA[i][ii]
            
            c_name_1 = 'feature' + str(i) + str(ii)
            c_name_2 = 'gain' + str(i) + str(ii)
            c_name_3 = 'split' + str(i) + str(ii)
            
            Re.columns = [c_name_1, c_name_2, c_name_3]
            Result_2 = pd.concat([Result_2, Re], axis = 1)
    Result_2.to_csv(save_path+'FI_p1.csv', index=False)
    
    # For Result_3
    Result_3 = pd.DataFrame([[start, end, end-start]], columns = ['Start', 'End', 'Time'])
    Result_3.to_csv(save_path+'time_p1.csv', index=False)
    
    # For Result_4
    Result_4 = np.stack(pred, axis=0)     
    np.save(save_path+'pred_result_p1.npy', Result_4)
