import argparse
import os
import shutil
import time
import sys

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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_adc as resnetmodeladc
import models.resnet as resnetmodel
from adcvariationindex import *
import adcdata
import numpy as np
import random
import math
import pdb

model_names = sorted(name for name in resnetmodeladc.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnetmodeladc.__dict__[name]))
model_names += sorted(name for name in resnetmodel.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnetmodel.__dict__[name]))
print('Available models:')
print(*model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--gpu', help = 'gpu-available', default = '2')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--pretrainedmodel', '-pm', default='resnet20_adc.th')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_adc',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
# parser.add_argument('--adc', help='adc model: ss, cco, ideal', default='noadc')
parser.add_argument('-adcb','--adc_bits', default=7, help='number of adc bits', type=int)
parser.add_argument('-adcfb','--adc_f_bits', default=5, help='number of adc fractional bits', type=int)
parser.add_argument('-adcbs','--adc_bit_scale', default=1.0,help='value corresponding to nth bit: adcbs*2**n-1', type=float)
parser.add_argument('-adcd','--adcdata', default='noadc', help='adc characterisitcs virtuoso dataset')
parser.add_argument('-adci','--adcindex', default=0, help='corner or mc index, 0 for nominal in all cases', type=int)
parser.add_argument('--wbits', default=7, type=int)
parser.add_argument('--wfbits', default=7, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=60, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-model-dir', dest='save_model_dir',
                    help='The directory to save the loading trained models',
                    default='saved_models', type=str)
parser.add_argument('--adcvat', help='ADC Variation aware training',
                    default=False, type=bool)
parser.add_argument('--adcms', help='ADV VAT random allocation manual seed',
                    default=42, type=int)
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True  #schange
    torch.backends.cudnn.benchmark = False     #schange  

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)  

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    #https://pytorch.org/docs/stable/notes/randomness.html
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    
    if (args.arch=='resnet20'):
        print('Train Model Details: {}'.format(resnetmodel.__dict__[args.arch].__doc__))
        model = torch.nn.DataParallel(resnetmodel.__dict__[args.arch]())
    elif (args.arch=='resnet20_adc'):
        print('Train Model Details: {}'.format(resnetmodeladc.__dict__[args.arch].__doc__))
        print('ADC Characteristics: {}'.format(adcdata.__dict__[args.adcdata].__doc__))
        print('Parameters:')
        print(' '*11, 'ADC Bits:', args.adc_bits)
        print(' '*11, 'ADC Fractional Bits:', args.adc_f_bits)
        print(' '*11, 'ADC Bit Scale:', args.adc_bit_scale)
        print(' '*11, 'ADC Corner/Index:', args.adcindex)
        print(' '*11, 'Weight Bits:', args.wbits)
        print(' '*11, 'Weight Fractional Bits:', args.wfbits)
        print(' '*11, 'Max ADC Out:', 2**(args.adc_bits-args.adc_f_bits)-1/2**(args.adc_f_bits))
        print(' '*11, 'Max Weight:', 2**(args.wbits-args.wfbits)-1/2**(args.wfbits))
        # print(' '*11+'-'*25)
        # print(' '*11, 'Weight Noise Std. Dev.:', args.weight_noise_std)
        print(' '*11+'-'*25)
        print(' '*11+' ADC VAT: ', args.adcvat)
        print(' '*11+'-'*25)
        if (args.adcvat):
            adc_char_list = []
            for i in range(200):
                adc_char_temp = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, i)
                adc_char_list.append(adc_char_temp)
            adc_char = torch.stack(adc_char_list)
            adc_char = get_var_adc_char(adc_char, args.adcms)
        else:
            adc_char = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, args.adcindex)

        adc_char = adc_char.cuda()
        nthlevels = adc_char.size(-1)
        print('Number of thresold levels:', nthlevels)
        print('ADC Bits: {}'.format(int(math.log2(nthlevels+1))))
        model = torch.nn.DataParallel(resnetmodeladc.__dict__[args.arch](adc_char, args.adc_f_bits, 
                                                                         args.adc_bit_scale, args.wbits, args.wfbits))
        
    # print('Dict',resnetmodel.__dict__[args.arch]())

    model.cuda()

    print('Loading model from:', os.path.join(args.save_model_dir, args.pretrainedmodel))
    pretrained_model=torch.load(os.path.join(args.save_model_dir, args.pretrainedmodel))
    print('Pretrained model training accuracy:', pretrained_model['best_prec1'])
    model.load_state_dict(pretrained_model['state_dict'])
    print('Pretrained model parameters loaded')    

    ##
    # print(pretrained_model['state_dict'].keys())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    validate(val_loader, model, criterion)

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

  
    
    return top1.avg, losses.avg, top1.avg 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    global plotter
#    plotter = utils.VisdomLinePlotter(env_name ='CIFAR10_ResNet20')    

    main()