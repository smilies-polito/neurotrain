import logging
import shutil
import os
import random

import torch
import numpy as np
import torch.nn.functional as F

def setup_logging(log_file='log.txt'):
    """Setup logging configuration"""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",   
                        datefmt="%Y-%m-%d %H:%M:%S",  
                        filename=log_file,
                        filemode='w')  

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), os.path.join(save_path, 'best.pth.tar'))
        

def lr_scheduler(optimizer, epoch, lr_decay_epoch=50, decay_factor=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor
    return optimizer

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def reproducible_config(seed=1234, is_cuda=False):
    """Some configurations for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(seed)

def average_pooling_through_time(x, time_window):
    for step in range(time_window):
        if step == 0:
            output = F.avg_pool2d(x[step], 2)
            output_return = output.clone()
        else:
            output = F.avg_pool2d(x[step], 2)
            output_return = torch.cat((output_return, output), dim=0)
    return output_return.view(-1, *output.size())