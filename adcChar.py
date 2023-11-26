import torch

# ADC Characteristics:ReLU_adc
# def ADC_Char(ADCIn, Interpolator):
#     Tramp = 50e-9                       # ramp time
#     f_clk = 128/Tramp                # 1 GHz clock frequency

#     torch.nn.functional.relu(ADCIn, inplace=True)
#     ADCIn = ADCIn.cpu()
#     ADCIn = ADCIn.detach().numpy()
#     OutComp = Interpolator(ADCIn)
#     OutComp = torch.from_numpy(OutComp)
#     OutComp = OutComp.to(torch.float32)
#     OutComp = OutComp.cuda()
#     OutCount = OutComp*f_clk
#     OutADC = torch.floor(OutCount)
#     return OutADC

def ADC_Char(ADCIn, Interpolator):
    Tramp = 50e-9                       # ramp time
    f_clk = 128/Tramp                # 1 GHz clock frequency

    # torch.nn.functional.relu(ADCIn, inplace=True)
    ADCIn = torch.where(ADCIn<2e-8, 2e-8, ADCIn)
    ADCIn = ADCIn.cpu()
    ADCIn = ADCIn.detach().numpy()
    OutFreq = Interpolator(ADCIn)
    OutCount = 128 * OutFreq / (7.157237e+08)
    OutCount = torch.from_numpy(OutCount)
    OutCount = OutCount.to(torch.float32)
    OutCount = OutCount.cuda()    
    OutADC = torch.floor(OutCount)
    return OutADC