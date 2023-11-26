from MyWorks.dataInterpols import InterpolF
from MyWorks.adcChar import ADC_Char

# def ReLU_adc (In, ReLURange):
#     ReLUIn = In * (1e-5) / ReLURange
#     ADCOut = ADC_Char(ReLUIn, InterpolF)
#     ReLUOut = ADCOut * ReLURange / 127
#     return ReLUOut

# class ReLU_adc(torch.nn.Module):
#     def __init__(self):
#         """
#         In the constructor we instantiate four parameters and assign them as
#         member parameters.
#         """
#         super().__init__()
#         self.Range = 25
#         self.ADCMax = 1e-5

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         return ADC_Char((x * self.ADCMax / self.Range), InterpolF) * self.Range / 127

#     # def string(self):
#     #     """
#     #     Just like any class in Python, you can also define custom method on PyTorch modules
#     #     """
#     #     return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

import torch
import torch.nn as nn
  
class ReLU_adc(nn.Module):
    def __init__(self):
        super(ReLU_adc, self).__init__()
  
    def forward(self, x):
        return ADC_Char((x * (9.920000e-06) / 25), InterpolF) * 25 / 127