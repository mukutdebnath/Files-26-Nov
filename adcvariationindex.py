import numpy as np
import torch

def get_var_adc_char(adc_chars, manual_seed, mc_range):  # adc_chars of size [200, 127] for 7 bit
    print(' '*11,'ADC VAT characteristic manual seed:', manual_seed)
    print(' '*11+'-'*25)
    torch.manual_seed(manual_seed)
    random_indices_1 = torch.randint(0, mc_range, (16, 32, 32))
    random_indices_2 = torch.randint(0, mc_range, (32, 16, 16))
    random_indices_3 = torch.randint(0, mc_range, (64, 8, 8))
    adc_char_1 = adc_chars[random_indices_1]
    adc_char_2 = adc_chars[random_indices_2]
    adc_char_3 = adc_chars[random_indices_3]
    return [adc_char_1, adc_char_2, adc_char_3]

if __name__=='__main__':
    result=get_var_adc_char(torch.randn((200,127)))
    print(result.shape)