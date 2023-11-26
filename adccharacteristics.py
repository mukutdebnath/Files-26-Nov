import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(root_dir, "test")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "Datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, test_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

import pandas
import numpy as np
from scipy import interpolate

def get_ss_lin_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/SS_ADC_10u.csv')
    data['delay_switch'] = data['delay_switch'] - data['delay_switch'].min()
    dataInMin = data['Iinput'].min()
    dataInMax = data['Iinput'].max()
    t = data['delay_switch'].max() - data['delay_switch'].min()

    dict_corner = {
        0: 'nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    print('ADC Characteristics: Linear SS ADC, Corner: {}'.format(dict_corner[adc_index]))
    data = data[data['Corner']==dict_corner[adc_index]]
    x, y = data['Iinput'], data['delay_switch']
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, t)

def get_ss_nonlin_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Data_10thJuly.csv')
    data['delay'] = data['delay'] - 60e-9
    dataInMin = data['Iinput'].min()
    dataInMax = data['Iinput'].max()
    t = 128e-9
    print('ADC Characteristics: Non Linear SS ADC, Corner: {}'.format(adc_index))
    x, y = data['Iinput'], data['delay']
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, t)

def get_cco_10u_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_char_all_corners.csv')
    freq_max = data.loc[99].max()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()
    dict_corner = {
        0: 'Nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    print('ADC Characteristics: Linear 10u CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_10u_adc_mc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_ADC_10u_MC.csv')
    freq_max = data['Freq'].max()
    dataInMin = data['i'].min()
    dataInMax = data['i'].max()
    print('ADC Characteristics: Linear 10u CCO ADC MC, MC Index: {}'.format(adc_index))
    x, y = data[data['mc_iteration']==adc_index+1]['i'],data[data['mc_iteration']==adc_index+1]['Freq']
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_nonlin_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_nonlin.csv')
    dict_char = {
        0: ('2.5u range', 'Iin2u', 'F2u'),
        1: ('10u range', 'Iin10u', 'F10u')
    }
    print('ADC Characteristics: Non Linear CCO ADC with {}'.format(dict_char[adc_index][0]))
    dataInMin = data[dict_char[adc_index][1]].min()
    dataInMax = data[dict_char[adc_index][1]].max()
    freq_max = data[dict_char[adc_index][2]].max()

    x, y = data[dict_char[adc_index][1]], data[dict_char[adc_index][2]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)


def get_cco_ibias_calib(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_ADC_Calib.csv')
    freq_max = data.loc[28].max()
    # breakpoint()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()

    dict_corner = {
        0: 'f_FF',
        1: 'f_SS'
    }
    print('ADC Characteristics: Ibias Calib CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_inckt3_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Incircuit_calib_char_for_cco_atallcorners_more_matching.csv')
    freq_max =data.loc[19].max()
    dataInMin = data['Freq X'].min()
    dataInMax = data['Freq X'].max()

    dict_corner = {
        0: 'Nom',
        1: 'FF',
        2: 'FS',
        3: 'SF',
        4: 'SS'
    }
    print('ADC Characteristics: Incircuit Calib 3 CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Freq X'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_inckt2_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Incircuit_18_bits_calib_char_for_cco_atallcorners.csv')
    freq_max =data.loc[19].max()
    dataInMin = data['INPUT I'].min()
    dataInMax = data['INPUT I'].max()

    dict_corner = {
        0: 'TT',
        1: 'FF',
        2: 'FS',
        3: 'SF',
        4: 'SS'
    }
    print('ADC Characteristics: Incircuit Calib 2 CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['INPUT I'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_inckt_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Incircuit_calib_char_for_cco_atallcorners.csv')
    freq_max =data.loc[19].max()
    dataInMin = data['Freq X'].min()
    dataInMax = data['Freq X'].max()

    dict_corner = {
        0: 'Freq Y',
        1: 'Freq Y.1',
        2: 'Freq Y.2',
        3: 'Freq Y.3',
        4: 'Freq Y.4'
    }
    print('ADC Characteristics: Incircuit Calib CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Freq X'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_inckt3_adc_mc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/MC_of_calibrated_cco_good_accuracy.csv')
    freq_max = data['FREQ'].max()
    dataInMin = data['I_input'].min()
    dataInMax = data['I_input'].max()
    print('ADC Characteristics: Incircuit 3 Calib CCO ADC MC, MC Index: {}'.format(adc_index))
    mc=adc_index+1
    datatemp=data[data['mc_iteration']==mc]
    x, y = datatemp['I_input'], datatemp['FREQ']
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_inckt3_adc_mc_wc_cal(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/calibrated_worstcase_MC_cco_good_accuracy.csv')
    freq_max = 455175207.849576
    dataInMin = data['18_after_calib X'].min()
    dataInMax = data['18_after_calib X'].max()
    dict_corner = {
        0: '18_after_calib Y',
        1: '24_after_calib Y',
        2: '198_after_calib Y',
        3: '169_after_calib Y',
        4: '188_after_calib Y'
    }
    print('ADC Characteristics: Incircuit 3 Calib CCO ADC MC worst case, MC Index: {}'.format(dict_corner[adc_index]))
    x, y = data['18_after_calib X'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_thermo_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Thermometer_Calibrated_CCO_2.5uA.csv')
    freq_max =data.loc[9].max()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()

    dict_corner = {
        0: 'TT 27C',
        1: 'SS 150C',
        2: 'SF 40C',
        3: 'FS 40C',
        4: 'FF -40C'
    }
    print('ADC Characteristics: Thermometer Calib CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)

    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_lVDD1V_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/low_VDD_CCO_1V.csv')
    freq_max = data.loc[28].max()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()
    dict_corner = {
        0: 'Nom',
        1: 'SS',
        2: 'SF',
        3: 'FS',
        4: 'FF'
    }
    print('ADC Characteristics: Low VDD 1V CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_lVDD0p9V_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/VDD_0.9V_2.csv')
    freq_max = 7.038353e+08
    if (adc_index==4):
        dataInMin = data['In FF'].iloc[0:9].astype(float).min()
        dataInMax = data['In FF'].iloc[0:9].astype(float).max()
    else:
        dataInMin = data['In Nom'].min()
        dataInMax = data['In Nom'].max()
    
    dict_corner = {
        0: 'Nom',
        1: 'SS',
        2: 'SF',
        3: 'FS',
        4: 'FF'
    }
    print('ADC Characteristics: Low VDD 0.9V CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    if (adc_index==4):
        x, y = data['In '+dict_corner[adc_index]].iloc[0:9].astype(float), data['Out '+dict_corner[adc_index]].iloc[0:9].astype(float)
    else:
        x, y = data['In '+dict_corner[adc_index]], data['Out '+dict_corner[adc_index]]        
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_lVDD0p8V_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/low_VDD_CCO_0not8V.csv')
    freq_max = data.loc[28].max()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()
    dict_corner = {
        0: 'Nom',
        1: 'SS',
        2: 'SF',
        3: 'FS',
        4: 'FF'
    }
    print('ADC Characteristics: Low VDD 0.8V CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_lVDD0p7V_adc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/low_VDD_CCO_0not7V.csv')
    freq_max = data.loc[28].max()
    dataInMin = data['Iin'].min()
    dataInMax = data['Iin'].max()
    dict_corner = {
        0: 'Nom',
        1: 'SS',
        2: 'SF',
        3: 'FS',
        4: 'FF'
    }
    print('ADC Characteristics: Low VDD 0.7V CCO ADC, Corner: {}'.format(dict_corner[adc_index]))
    x, y = data['Iin'], data[dict_corner[adc_index]]
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_cco_lVDD1V_adc_mc(adc_index):
    data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/VDD_1V_mc.csv')
    data = data[data['FREQ']!='eval err']
    data['FREQ']=data['FREQ'].astype(float)
    freq_max=data['FREQ'].max()
    dataInMin = data['i'].min()
    dataInMax = data['i'].max()
    print('ADC Characteristics: Low VDD = 1V CCO ADC MC, MC Index: {}'.format(adc_index))
    mc=adc_index+1
    datatemp=data[data['mc_iteration']==mc]
    x, y = datatemp['i'], datatemp['FREQ']
    InterpolF = interpolate.interp1d(x, y)
    return InterpolF, (dataInMin, dataInMax, freq_max)

def get_adc(adc, adc_index):
    if(adc=='linss'):
        return get_ss_lin_adc(adc_index)
    elif(adc=='lin10ucco'):
        return get_cco_10u_adc(adc_index)
    elif(adc=='lin10uccomc'):
        return get_cco_10u_adc_mc(adc_index)
    elif(adc=='nonlinss'):
        return get_ss_nonlin_adc(adc_index)
    elif(adc=='nonlincco'):
        return get_cco_nonlin_adc(adc_index)
    elif(adc=='ibiascco'):
        return get_cco_ibias_calib(adc_index)
    elif(adc=='incktcco3'):
        return get_cco_inckt3_adc(adc_index)
    elif(adc=='incktcco2'):
        return get_cco_inckt2_adc(adc_index)
    elif(adc=='incktcco'):
        return get_cco_inckt_adc(adc_index)   
    elif(adc=='incktcco3mc'):
        return get_cco_inckt3_adc_mc(adc_index)  
    elif(adc=='incktcco3mcwccal'):
        return get_cco_inckt3_adc_mc_wc_cal(adc_index) 
    elif(adc=='thermocco'):
        return get_cco_thermo_adc(adc_index)
    elif(adc=='lvdd1vcco'):
        return get_cco_lVDD1V_adc(adc_index)
    elif(adc=='lvdd0p9vcco'):
        return get_cco_lVDD0p9V_adc(adc_index)
    elif(adc=='lvdd0p8vcco'):
        return get_cco_lVDD0p8V_adc(adc_index)
    elif(adc=='lvdd0p7vcco'):
        return get_cco_lVDD0p7V_adc(adc_index)
    elif(adc=='lvdd1vccomc'):
        return get_cco_lVDD1V_adc_mc(adc_index)
    # elif(adc=='lvdd0p8vccomc'):
    #     return get_cco_lVDD0p8V_adc_mc(adc_index)
    # elif(adc=='lvdd0p7vccomc'):
    #     return get_cco_lVDD0p7V_adc_mc(adc_index)
    else:
        return 0, (0,0,0)