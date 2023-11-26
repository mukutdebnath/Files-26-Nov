import pandas
import numpy as np
from scipy import interpolate
import torch
from sklearn.linear_model import LinearRegression

__all__=['ss_lin','ss_nl_wo_cal_mc', 'cco_lin', 'ideal', 'cco_aut_cal', 'cco_lvd_p8v_mc', 
         'cco_mc', 'noadc', 'ss_nl_cal', 'vco_mc', 'ss_64_mc', 'ss_vmode_nicap']

def get_thresholds(x, y): 
    # takes analog x and quantized y (full scaled mapping according to adc output)
    # gives output of the threshold levels where the adc digital code changes
    # zero offset is used to provide any initial count value if required
    # make sure to give fully scaled x and y values, 
    # such as x: analog 0 to 10
    # y: quantized 0 to 10

    op_diff = np.diff(y)
    change_idx = np.nonzero(op_diff)[0]   # [0] is used as return is [n,1] in shape
    th_levels = x[change_idx+1]    # +1 as indices returned are those after which the output y changes
    return th_levels

def ss_lin(adc_bits, adc_f_bits, adc_index):
    "SS ADC Linear Ideal Ramp"
    data=pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/SS_ADC_10u.csv')
    data['delay_switch']=data['delay_switch']-data['delay_switch'].min()
    fclk=1/(data['delay_switch'].max()-data['delay_switch'].min()) # most ideal count evaluation
    corner_dict={
        0: 'nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    print(' '*11,'ADC Corner Name:', corner_dict[adc_index])
    print(' '*11+'-'*25)
    data=data[data['Corner']==corner_dict[adc_index]]
    If=interpolate.interp1d(data['Iinput'],data['delay_switch'])
    x=np.arange(data['Iinput'].min(), data['Iinput'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits*fclk)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)
    
def ss_nl_cal(adc_bits, adc_f_bits, adc_index):
    "SS ADC Real Ramp"
    data=pandas.read_csv('../Simdata/ADC_Char_With_Cal.csv')
    corner_dict={
        0: 'Nominal',
        1: 'ss150',
        2: 'ff_40',
        3: 'sf27',
        4: 'fs27'
    }
    print(' '*11,'ADC Corner Name:', corner_dict[adc_index])
    print(' '*11+'-'*25)
    if (adc_index==4):
        If=interpolate.interp1d(data['Iinput'].drop(13),data[corner_dict[adc_index]].drop(13).astype(float))
        x=np.arange(data['Iinput'].drop(13).min(), data['Iinput'].drop(13).max(), 1e-10)
    else: 
        If=interpolate.interp1d(data['Iinput'],data[corner_dict[adc_index]])
        x=np.arange(data['Iinput'].min(), data['Iinput'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*1e9)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)

def cco_lin(adc_bits, adc_f_bits, adc_index):
    data=pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_char_all_corners.csv')
    fmax=data.loc[99].max()
    corner_dict={
        0: 'Nom',
        1: 'C0',
        2: 'C1',
        3: 'C2',
        4: 'C3'
    }
    print(' '*11,'ADC Corner Name:', corner_dict[adc_index])
    print(' '*11+'-'*25)
    If=interpolate.interp1d(data['Iin'], data[corner_dict[adc_index]])
    x=np.arange(data['Iin'].min(), data['Iin'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    if (ycount[0]>0):
        zeros=[0]*ycount[0]
        ycount=np.concatenate(np.array(zeros), ycount)
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    return torch.tensor(th_levels)

def cco_mc(adc_bits, adc_f_bits, adc_index):
    "CCO ADC Monte Carlo"
    data=pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_ADC_10u_MC.csv')
    fmax=data['Freq'].max()
    data_mc=data[data['mc_iteration']==adc_index+1]
    If=interpolate.interp1d(data_mc['i'], data_mc['Freq'])
    x=np.arange(data['i'].min(), data['i'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)   
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)

def cco_aut_cal(adc_bits, adc_f_bits, adc_index):
    "CCO ADC Automatic Calibration Aug 17"
    data=pandas.read_csv('../Simdata/automatic_calib_CCO.csv')
    fmax=data.iloc[19].max()
    corner_dict={
        0: 'TT',
        1: 'SS',
        2: 'SF',
        3: 'FS',
        4: 'FF'
    }
    # print(' '*11,'ADC Corner Name:', corner_dict[adc_index])
    # print(' '*11+'-'*25)
    If=interpolate.interp1d(data['Iin'],data[corner_dict[adc_index]])
    x=np.arange(data['Iin'].min(), data['Iin'].max(), 1e-10)
    if (x[-1] >= data['Iin'].max()):
        x=np.delete(x, -1)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)   
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)
    
def cco_lvd_p8v_mc(adc_bits, adc_f_bits, adc_index):
    "CCO Low VDD ADC MC"
    data=pandas.read_csv('../Simdata/VDD_0p8V_mc.csv')
    data=data[data['i'] > data['i'].min()]
    data['FREQ'] = data['FREQ'].astype(float)
    fmax=data['FREQ'].max()
    data_mc=data[data['mc_iteration']==adc_index+1]
    If=interpolate.interp1d(data_mc['i'], data_mc['FREQ'])
    x=np.arange(data['i'].min(), data['i'].max(), 1e-10)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)   
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)

def ss_nl_wo_cal_mc(adc_bits, adc_f_bits, adc_index):
    "Real SS ADC Characteristics without Calibration"
    data=pandas.read_csv('../Simdata/ADC_Char_Without_Cal.csv')
    data=data[~data['Z2'].str.contains('eval err')]
    data['Z2']=data['Z2'].astype(float)
    data.loc[data['temperature']==-40, 'mc_iteration'] += 50
    data=data[data['mc_iteration']==adc_index+1]
    If=interpolate.interp1d(data['Iinput'],data['Z2'])
    x=np.arange(data['Iinput'].min(), data['Iinput'].max()-1e-6, 1e-10)
    y=If(x)
    ycount=np.floor(y*1e9)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)
    
def vco_mc(adc_bits, adc_f_bits, adc_index):
    "VCO ADC MC"
    data=pandas.read_csv('../Simdata/vco.csv')
    data['Vctrl']=data['Vctrl'] - data['Vctrl'].min()          # input offset correction
    fmax=data['Freq'].max()
    data_mc=data[data['mc_iteration']==adc_index+1]
    If=interpolate.CubicSpline(data_mc['Vctrl'], data_mc['Freq'])
    x=np.arange(data_mc['Vctrl'].min(), data_mc['Vctrl'].max()-1e-3, 1e-6)
    y=If(x)
    ycount=np.floor(y*2**adc_bits/fmax)
    if (ycount[-1]==128):
        ycount=np.delete(ycount,-1)
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)   
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)
    
def ss_64_mc(adc_bits, adc_f_bits, adc_index):
    "VCO MC Cal at 64"
    data=pandas.read_csv('../Simdata/MC_PLOTS_64.csv')
    If=interpolate.interp1d(data['I_input'], data['Unnamed: '+str(adc_index+1)])
    x=np.arange(data['I_input'].min(), data['I_input'].max(), 1e-9)
    y=If(x)
    ycount=np.floor(y*1e9)
    # if (ycount[-1]==128):
    #     ycount=np.delete(ycount,-1)
    mask = ycount != 128
    ycount = ycount[mask]
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    xnorm=x*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)
    
def ss_vmode_nicap(adc_bits, adc_f_bits, adc_index):
    "8th October Voltage mode SS ADC, non ideal cap"
    data = pandas.read_csv('../Simdata/8thOct_VMode_SSADC_Ch_Curves_Non_Ideal_Cap.csv')
    def correct_index(row):
        if row['Corner'] == 'C1':
            return row['mc_iteration'] + 70
        else:
            return row['mc_iteration']
    data['mc_iteration'] = data.apply(correct_index, axis=1)
    data = data[['Vin', 'mc_iteration', 'Delay']]
    # print(data)
    datax = data[data['mc_iteration']==adc_index+1]
    # print(datax)
    datax['Delay'] = pandas.to_numeric(datax['Delay'], errors='coerce')
    valid_data = datax[['Vin', 'Delay']].dropna()
    missing_data = datax[['Vin', 'Delay']][datax['Delay'].isna()]

    model = LinearRegression()
    model.fit(valid_data[['Vin']], valid_data['Delay'])

    predicted_y = model.predict(datax[['Vin']])
    datax['Delay'] = predicted_y

    datax.sort_values(by='Vin', inplace=True)
    datax.reset_index(drop=True, inplace=True)

    # print(datax)
    datax['Vin'] = datax['Vin'] - data['Vin'].min()
    datax['Delay'] = datax['Delay'] - 1e-9
    # print(datax)

    If=interpolate.interp1d(datax['Vin'], datax['Delay'])
    x=np.arange(datax['Vin'].min(), datax['Vin'].max(), 1e-5)
    y=If(x)
    ycount=np.floor( 128*y/10e-9 )
    # if (ycount[-1]==128):
    #     ycount=np.delete(ycount,-1)
    mask = ycount < 128
    ycount = ycount[mask]
    
    adc_range = 2**(adc_bits-adc_f_bits)-1/2**(adc_f_bits)
    xnorm=x[0:len(ycount)]*adc_range/x.max()
    th_levels=get_thresholds(xnorm, ycount)
    if (ycount[0]>0):
        th_levels=np.concatenate((np.array([0]*int(ycount[0])), th_levels))
    if (2**adc_bits-1 != len(th_levels)):
        th_levels=np.concatenate((th_levels, 
                                 np.array([2**(adc_bits-adc_f_bits)]*(2**adc_bits-1-len(th_levels)))))
    if (len(th_levels) != 2**adc_bits - 1):
        print(th_levels)
        print(len(th_levels))
        raise Exception('Number of threshold levels does not match ', 2**adc_bits - 1, '. Check code again.')
    else:
        return torch.tensor(th_levels)




def ideal(adc_bits, adc_f_bits, adc_index):
    "Ideal ADC, uniform quantization"
    return torch.tensor(np.arange(1,2**adc_bits,1)/2**adc_f_bits)

def noadc(adc_bits, adc_f_bits, adc_index):
    "No ADC, Ideal Activation: ADC(x) = x"
    return 0

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-ai','--adcindex', default=0, type=int)
    args=parser.parse_args()
    adc_bits=(7,0)    
    out=ss_vmode_nicap(adc_bits=adc_bits[0], adc_f_bits=adc_bits[1], adc_index=args.adcindex)
    print(out)
    print(out.shape)