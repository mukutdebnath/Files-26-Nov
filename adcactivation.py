import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# threshold has shape [1, #adc levels]
# x.unsqueeze(-1) has shape [shape, 1]
# the comparison gives output of shape  [shape, #adc_levles]
# sum is taken of last dimension i.e., -1 which gives the number of times the 
# comparison output is True, which gives the count in integer,
# convert it to float.
# scale down to fractional bits

# for fixed variations, the adc_char matrix needs to be of shape [imsize,#adc_thresholds]
# while input.iunsqueeze(-1) will be of shape [batch_size, imsize, 1]
# verified that the comparision will give imsize different matching, and hence include the fixed 
# adc variations as desired.
# the adc_char needs to be exported from the simdata in this manner

def threshold_quantize(adc_f_bits, adc_char, bit_scale):
    class tqn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # if (adc_char.dim()==1 or adc_char.size(0)==input.size(1)):
            #     quant_op_int=torch.sum(input.unsqueeze(-1) >= adc_char, dim=-1).float()
            # else:
            #     adc_char_new = adc_char.view(-1, input.size(1), input.size(2), input.size(3), adc_char.size(3))[0]
            #     quant_op_int = torch.sum(input.unsqueeze(-1) >= adc_char_new, dim=-1).float()
            if isinstance(adc_char, torch.Tensor) and adc_char.size(0) != 200:
                quant_op_int=torch.sum(input.unsqueeze(-1) >= adc_char, dim=-1).float()
            elif isinstance(adc_char, torch.Tensor) and (adc_char.size(0)==200):
                random_indices = torch.randint(0, 200, (input.size(0), input.size(1), 
                                                          input.size(2), input.size(3)))
                adc_char_1 = adc_char[random_indices]
                quant_op_int=torch.sum(input.unsqueeze(-1) >= adc_char_1, dim=-1).float()
            else:
                for char in adc_char:
                    if (char.size()[0:-1]==input.size()[1:]):
                        quant_op_int=torch.sum(input.unsqueeze(-1) >= char, dim=-1).float()
                        break
            quant_op=quant_op_int/2**adc_f_bits
            op=bit_scale*(quant_op)
            return op
        
        @staticmethod
        def backward(ctx, grad_output):
            grad_input=grad_output.clone()
            return grad_input
        
    return tqn().apply  


class ADCActivation(nn.Module):
    def __init__(self, adc_f_bits, adc_char, bit_scale):
        super(ADCActivation, self).__init__()
        self.adc_f_bits=adc_f_bits
        self.adc_char=adc_char
        self.bit_scale=bit_scale
        self.tqn_function=threshold_quantize(self.adc_f_bits, self.adc_char, self.bit_scale)

    def forward(self, x):
        if isinstance(self.adc_char, int): # no adc
            return x
        else:
            activation=torch.clamp(x, 0, 2**(7-self.adc_f_bits)-1/2**(self.adc_f_bits))
            return self.tqn_function(activation)

if __name__=="__main__":
    from adcdata import *
    adc_bits=(7,5)
    adc_index=0
    bit_scale=1
    # adc_charac_ss=cco_lin(adc_bits[0],adc_bits[1], adc_index)
    # adc_charac_ss=adc_charac_ss.cuda()
    # adc_func_ss=threshold_quantize(adc_bits[1], adc_charac_ss, bit_scale)
    # adc_charac_ideal=ideal(adc_bits[0],adc_bits[1], adc_index)
    # adc_charac_ideal=adc_charac_ideal.cuda()
    # adc_func_ideal=threshold_quantize(adc_bits[1], adc_charac_ideal, bit_scale)
    # adcnet=ADCActivation(adc_f_bits=adc_bits[1], adc_characteristics=adc_charac, zero_offset=zero_off,bit_scale=bit_scale)
    # print('Ideal:\n',adc_charac_ideal)
    # print('SS:\n',adc_charac_ss)
    x=torch.from_numpy(np.arange(0, 2**(adc_bits[0]-adc_bits[1])-1/2**(adc_bits[1]), 1e-4))
    x=x.cuda()
    plt.figure
    for i in range(0, 140):
        adc_charac=ss_vmode_nicap(adc_bits[0],adc_bits[1], i)
        adc_charac=adc_charac.cuda()
        adc_func=threshold_quantize(adc_bits[1], adc_charac, bit_scale)
        plt.plot(x.cpu().numpy(), adc_func(x).cpu().numpy(), label=i, linewidth=0.75)
    # plt.legend()
    plt.show()
    plt.close()