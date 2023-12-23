import pandas as pd
import numpy as np
import json

def data_OV_proc(file, resample_time):
    with open(file) as f:
      data = json.load(f)
    details = data['detail']
    df1 = pd.DataFrame(details)
    df1.index = pd.to_datetime(df1.dateTime)
    df1['oil'] = df1['oil'].map(lambda x: x+273.15)
    df1['siemAnode'] = df1['siemAnode'].map(lambda x: x*1000)
    df2 = df1[['oil', 'siemAnode']]
    df_selected = df2.resample(resample_time).mean() 
    
    df_selected['oil'] = df_selected['oil'].interpolate()
    df_selected['siemAnode'] = df_selected['siemAnode'].interpolate()
    df_selected.columns = ['Oil','Voltage']
    return df_selected

def data_TE_proc(file, resample_time):
    with open(file) as f:
      data_tss = json.load(f)
    details_tss0=data_tss['detail'][1]
    df1 = pd.DataFrame(details_tss0['tubeData'])
    df1.index = pd.to_datetime(df1.dateTime)
    df1['tubeKws'] = df1['tubeKws'].map(lambda x: x*3600000)
    df2 = df1[['tubeScanSeconds', 'tubeKws']]
    df_selected = df2.diff().resample(resample_time).sum()
    df_selected.replace(0, np.nan, inplace=True)

    df_selected['tubeScanSeconds'] = df_selected['tubeScanSeconds'].interpolate()
    df_selected['tubeKws'] = df_selected['tubeKws'].interpolate()
    df_selected.columns = ['Time', 'Energy']
    return df_selected

def data_TE_proc2(file, resample_time):
    with open(file) as f:
      data_tss = json.load(f)
    details_tss0=data_tss['detail'][0]
    df1 = pd.DataFrame(details_tss0['tubeData'])
    df1.index = pd.to_datetime(df1.dateTime)
    df1['tubeKws'] = df1['tubeKws'].map(lambda x: x*3600000)
    df2 = df1[['tubeScanSeconds', 'tubeKws']]
    df_selected = df2.diff().resample(resample_time).sum()
    df_selected.replace(0, np.nan, inplace=True)
    
    df_selected['tubeScanSeconds'] = df_selected['tubeScanSeconds'].interpolate()
    df_selected['tubeKws'] = df_selected['tubeKws'].interpolate()
    df_selected.columns = ['Time', 'Energy']
    return df_selected

def data_A_proc(file, resample_time):
    with open(file) as f:
      data_arc = json.load(f)
    details_arc=data_arc['detail']
    df_arc = pd.DataFrame(details_arc)
    df_arc.index = pd.to_datetime(df_arc.dateTime)
    df_arc['Arc'] = df_arc.dailyArcs.apply(lambda x : 1 if x > 0 else 0)

    df_A = df_arc[['Arc']].resample(rule='d').mean()
    df_A = df_A.fillna(0)
    df2 = df_A[['Arc']].resample(resample_time).mean()
    df2 = df2.ffill() 
    df2.columns = ['Label']
    return df2


def CT_data_loader(resample_time):
    CT1_T = data_OV_proc('data/CT/CT2_DEV000000008/DEV000000008_CtTubeTemp_365', resample_time)
    CT1_TE = data_TE_proc('data/CT/CT2_DEV000000008/DEV000000008_TubeScanSeconds_365', resample_time)
    CT1_A = data_A_proc('data/CT/CT2_DEV000000008/DEV000000008_CtTubeArc_365', resample_time)
    CT1 = pd.concat([CT1_T, CT1_TE, CT1_A], axis=1, join='inner')
    CT1.index = range(len(CT1))
    CT1['Machine'] = 1

    CT2_T = data_OV_proc('data/CT/CT3_DEV000000011/DEV0000000011_CtTubeTemp_365', resample_time)
    CT2_TE = data_TE_proc2('data/CT/CT3_DEV000000011/DEV0000000011_TubeScanSeconds_365', resample_time)
    CT2_A = data_A_proc('data/CT/CT3_DEV000000011/DEV0000000011_CtTubeArc_365', resample_time)
    CT2 = pd.concat([CT2_T, CT2_TE, CT2_A], axis=1, join='inner')
    CT2.index = range(len(CT2))
    CT2['Machine'] = 2

    CT3_T = data_OV_proc('data/CT/CT6_DEV000000017/DEV0000000017_CtTubeTemp_365', resample_time)
    CT3_TE = data_TE_proc2('data/CT/CT6_DEV000000017/DEV0000000017_TubeScanSeconds_365', resample_time)
    CT3_A = data_A_proc('data/CT/CT6_DEV000000017/DEV0000000017_CtTubeArc_365', resample_time)
    CT3 = pd.concat([CT3_T, CT3_TE, CT3_A], axis=1, join='inner')
    CT3.index = range(len(CT3))
    CT3.drop([0],inplace = True)
    CT3['Machine'] = 3

    CT = pd.concat([CT1, CT2, CT3], axis=0)
    CT = CT.ffill() 
    CT = CT.bfill() 
    CT['Current'] = CT.apply(lambda x: x['Energy'] / (x['Voltage']*x['Time']), axis=1)
    CT = CT[['Oil','Voltage','Current','Energy', 'Label', 'Machine']]
    return CT

def ai4i2020_data_loader():
    raw_data = pd.read_csv('data/ai4i2020/ai4i2020.csv')            
    data = raw_data[['Air temperature [K]',
                     'Process temperature [K]', 
                     'Rotational speed [rpm]', 
                     'Torque [Nm]',
                     'Tool wear [min]']]
    
    target = raw_data[['Machine failure']]
    data.columns = ['AT', 'PT', 'RS', 'T', 'TW']
    target.columns = ['Label']
        
    ai4i_data = pd.concat([data, target], axis=1)
    ai4i_data['Machine'] = 1
    return ai4i_data










