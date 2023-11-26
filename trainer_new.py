#https://github.com/akamaster/pytorch_resnet_cifar10
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
from adcvariationindex import get_var_adc_char
import adcdata
import numpy as np
import random
import math
import pdb
from models.quant_dorefa import *

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
parser.add_argument('--savemodel', '-sm', default='resnet20_adc.th')
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
parser.add_argument('--wbits', default=5, type=int)
parser.add_argument('--wfbits', default=5, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=60, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-model-dir', dest='save_model_dir',
                    help='The directory to save the final trained models',
                    default='saved_models', type=str)
# parser.add_argument('--save-every', dest='save_every',
#                     help='Saves checkpoints at every specified number of epochs',
#                     type=int, default=10)
parser.add_argument('--weight-noise-std-gamma', '-wnstd', dest='weight_noise_std_gamma',
                    help='additive weight noise standard deviation for training',
                    default=0.0, type=float)
parser.add_argument('--adcvat', help='ADC Variation aware training',
                    default=False, type=bool)
parser.add_argument('--adcvatf', help='ADC Fully Random Variation aware training',
                    default=False, type=bool)
parser.add_argument('--adcms', help='ADV VAT random allocation manual seed',
                    default=42, type=int)
best_prec1 = 0
args = parser.parse_args()

## https://github.com/uoguelph-mlrg/Cutout/blob/287f934ea5fa00d4345c2cccecf3552e2b1c33e3/util/cutout.py#L5
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def dec2bin_comps(decimal):
    binary_representation = bin(decimal)[2:]  # Convert to binary and remove '0b' prefix
    binary_components = [int(bit) * 2 ** (len(binary_representation) - i - 1) for i, bit in enumerate(binary_representation)]
    return binary_components

def get_sigma(mean):
  bin_components = dec2bin_comps(abs(mean))
  return sum([x ** 2 for x in bin_components])    

wt_quant=weight_quantize_fn(w_bit=args.wbits, wf_bit=args.wfbits)

def add_weight_noise(model, noise_std_gamma, wfbits):
    for name, param in model.named_parameters():
        if ('conv' in name or 'linear' in name) and 'weight' in name:
            noise_0_1 = torch.randn_like(param)                    # noise values with 0 mean 1 std dev
            weights_int = wt_quant(param.clone()) * (2**wfbits)               # if 2 bit weight, 2 frac bits, then 0001 = 0.25
            # print(weights_int.max())
            weights_var_unit = weights_int.to(dtype=torch.int).to('cpu').apply_(get_sigma)  # returns the squared sum of the binary elements squared
            weights_std_unit = torch.sqrt(weights_var_unit.to(param.device))                # returns the sq root of squared sum of binary elements
            weights_std_unit_gamma = weights_std_unit.to(param.device) * noise_std_gamma    # std dev value required
            weights_std = weights_std_unit_gamma/(2**wfbits)    # scaled to fractional bits req
            noise_0_std = torch.mul(noise_0_1, weights_std)     # noise with required std and 0 mean
            param.data += noise_0_std.to(param.device)          # weight updated to value from distributoion
                                                                # with weight as mean and calculated std

        # print(param)
        # if ('conv' in name or 'linear' in name) and 'weight' in name:
        #     noise = torch.randn_like(param) * noise_std * param.data / (2**(args.wbits-args.wfbits)-1/2**(args.wfbits))
        #     param.data += noise.to(param.device)
            # print('Noise added to', name)

def main():
    global args, best_prec1
    torch.backends.cudnn.deterministic = True  #schange
    torch.backends.cudnn.benchmark = False     #schange 

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
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
        print(' '*11+'-'*25)
        print(' '*11, 'Weight Noise Std. Dev. Gamma:', args.weight_noise_std_gamma)
        print(' '*11+'-'*25)
        print(' '*11+' ADC VAT: ', args.adcvat)
        print(' '*11+'-'*25)
        print(' '*11+' ADC VAT Full Random: ', args.adcvatf)
        print(' '*11+'-'*25)
        if (args.adcvat and not args.adcvatf):
            adc_char_list = []
            if (args.adcdata=='ss_nl_wo_cal'):
                mc_range=100
            elif (args.adcdata=='ss_vmode_nicap'):
                mc_range=140
            else:
                mc_range=200
            for i in range(mc_range):
                adc_char_temp = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, i)
                adc_char_list.append(adc_char_temp)
            adc_char = torch.stack(adc_char_list)
            adc_char = get_var_adc_char(adc_char, args.adcms, mc_range)
            adc_char = [char.cuda() for char in adc_char]
            if(adc_char[0].size(-1)==adc_char[1].size(-1)==adc_char[2].size(-1)):
                nthlevels = adc_char[0].size(-1)
            else:
                raise Exception('Some Error in extracting ADC Characteristics: Check adcvariationindex.py')
        elif (args.adcvatf and not args.adcvat):
            adc_char_list = []
            if (args.adcdata=='ss_nl_wo_cal'):
                mc_range=100
            elif (args.adcdata=='ss_vmode_nicap'):
                mc_range=140
            else:
                mc_range=200
            for i in range(mc_range):
                # print(i)
                adc_char_temp = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, i)
                adc_char_list.append(adc_char_temp)
            adc_char = torch.stack(adc_char_list)
            adc_char=adc_char.cuda()
            nthlevels=adc_char.size(-1)
        else:
            adc_char = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, args.adcindex)
            if not isinstance(adc_char, int):
                adc_char = adc_char.cuda()
                nthlevels = adc_char.size(-1) 
            else:
                nthlevels = 1        
        
        print('Number of thresold levels:', nthlevels)
        print('ADC Bits: {}'.format(int(math.log2(nthlevels+1))))
        model = torch.nn.DataParallel(resnetmodeladc.__dict__[args.arch](adc_char, 
                                                                         args.adc_f_bits, 
                                                                         args.adc_bit_scale, 
                                                                         args.wbits, 
                                                                         args.wfbits, 
                                                                         args.weight_noise_std_gamma))
    # print('Dict',resnetmodel.__dict__[args.arch]())
    model.cuda()
#    param = pytorch_count_params(model)
#    print('total no. of params:', param)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

#    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes = 1, length = 16)
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    #pdb.set_trace()

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

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], 
                                                        last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_result= train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        test_result = validate(val_loader, model, criterion)

        is_best = test_result[0] > best_prec1
        best_prec1 = max(test_result[0], best_prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_model_dir, args.savemodel))
        print('Best Accuracy:{}'.format(best_prec1))
#        pdb.set_trace()
    print('Saving the trained model as:', args.savemodel)

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # add weight noise
        # if (args.weight_noise_std_gamma != 0):
        #     add_weight_noise(model, args.weight_noise_std_gamma, args.wfbits)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
    return losses.avg, top1.avg 


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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, filename)

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







