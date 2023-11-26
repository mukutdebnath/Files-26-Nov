#https://github.com/akamaster/pytorch_resnet_cifar10
import argparse
import os
import sys
import shutil
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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils    # For visdom to plot the accuracy and loss plots
import numpy as np
import random
import src.config as cfg

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *

from models import resnet_adc as resnetmodeladc
from models import resnet_mvm as resnetmodelmvm

from quant_dorefa import *
from MyWorks import adcdata

model_names = sorted(name for name in resnetmodeladc.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnetmodeladc.__dict__[name]))
model_names += sorted(name for name in resnetmodelmvm.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnetmodelmvm.__dict__[name]))
print('Available models:')
print(*model_names)


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--gpu', help = 'gpu-available', default = '2')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--pretrainedmodel', '-pm', default='resnet20_adc.th')
parser.add_argument('--archpt', '-apt', metavar='ARCHPT', default='resnet20_adc',
                    choices=model_names,
                    help='pretrained model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32_adc)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_mvm',
                    choices=model_names,
                    help='test model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32_mvm)')
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
best_prec1 = 0

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

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True  #schange
    torch.backends.cudnn.benchmark = False     #schange    

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    #https://pytorch.org/docs/stable/notes/randomness.html
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    
    # model = torch.nn.DataParallel(resnetmodel.__dict__[args.arch](args.adc, args.adcbits))    
    print('Loading pretrained model from ',os.path.join(args.save_model_dir, args.pretrainedmodel))
    adc_char = adcdata.__dict__[args.adcdata](args.adc_bits, args.adc_f_bits, args.adcindex)
    # adc_char = adc_char.cuda()
    original_model = torch.nn.DataParallel(resnetmodeladc.__dict__[args.archpt](adc_char, args.adc_f_bits, 
                                                                         args.adc_bit_scale, args.wbits, args.wfbits))
    model=torch.nn.DataParallel(resnetmodelmvm.__dict__[args.arch]())
    # model.load_state_dict(torch.load(args.pretrainedmodel))
    pretrained_model=torch.load(os.path.join(args.save_model_dir, args.pretrainedmodel))
    print('Pretrained model training accuracy:', pretrained_model['best_prec1'])
    original_model.load_state_dict(pretrained_model['state_dict'])
    weights_conv = []
    weights_lin = []
    bias_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    count_conv=count_bn=count_linear=0

    wt_quant=weight_quantize_fn(w_bit=7, wf_bit=args.wfbits)
    for name, m in original_model.named_modules():
        # print(name, m)
        if 'fc' in name and 'quantize_fn' not in name:
            m.weight.data = wt_quant(m.weight.data)
        elif 'resconv' in name:
            if '.0' in name and 'quantize_fn' not in name:
                m.weight.data = wt_quant(m.weight.data)
        elif 'conv' in name and 'quantize_fn' not in name:
            m.weight.data = wt_quant(m.weight.data)
    
    for m in original_model.modules():
            if isinstance(m, nn.Conv2d):
                weights_conv.append(m.weight.data.clone())
                count_conv+=1
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_data.append(m.weight.data.clone())
                bn_bias.append(m.bias.data.clone())
                running_mean.append(m.running_mean.data.clone())
                running_var.append(m.running_var.data.clone())
                num_batches.append(m.num_batches_tracked.clone())
                count_bn+=1
            elif isinstance(m, nn.Linear):
                weights_lin.append(m.weight.data.clone())
                count_linear+=1

    i=j=k=0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, Conv2d_mvm)):
            m.weight.data = weights_conv[i]
            i = i+1
        #print(m.weight.data)
        #raw_input()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        elif isinstance(m, Linear_mvm) or isinstance(m, nn.Linear):
#        elif isinstance(m, nn.Linear):
            m.weight.data = weights_lin[k]
            # m.bias.data = bias_lin[k]
            k=k+1      
    
    if (i==count_conv and j==count_bn and k==count_linear):
        print('All pretrained parameters loaded successfully')
    else:
        print('Pretrained parameters not loaded correctly')

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

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    test_result = validate(val_loader, model, criterion)
    print('Test Accuracy:{}'.format(test_result[0]))

    # print('ADC: {}, Bits: {}'.format(args.adc, args.adcbits))


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







