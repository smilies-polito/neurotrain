import torch
import torch.nn as nn

from utils import average_pooling_through_time

class SVHNCNN(nn.Module):
    def __init__(self, params):
        super(SVHNCNN, self).__init__()
        self.encoding = params['encoding']
        self.learning_rule = params['learning_rule']

        if self.encoding == 'latency':
            first_layer = False
        else:
            first_layer = True

        if self.learning_rule == 'FELL':
            from local_linear_FELL import LocalLossBlockLinear
            from local_conv_FELL import LocalLossBlockConv
        elif self.learning_rule == 'BELL':
            from local_linear_BELL import LocalLossBlockLinear
            from local_conv_BELL import LocalLossBlockConv
        elif self.learning_rule == 'ELL':
            from local_linear_ELL import LocalLossBlockLinear
            from local_conv_ELL import LocalLossBlockConv
        else:
            raise Exception('Unrecognized learning rule.')

        if 'encoding' in params.keys():
            params.pop('encoding')
        if 'learning_rule' in params.keys():
            params.pop('learning_rule')
        if 'encoding' in params.keys():
            params.pop('encoding')

        self.conv1 = LocalLossBlockConv(ch_in=3, ch_out=64, kernel_size=3, stride=1, padding=1, dim_out=32, first_layer=first_layer, **params)
        self.conv2 = LocalLossBlockConv(ch_in=64, ch_out=64, kernel_size=3, stride=1, padding=1, dim_out=32, **params)
        
        self.conv3 = LocalLossBlockConv(ch_in=64, ch_out=128, kernel_size=3, stride=1, padding=1, dim_out=16, **params)
        self.conv4 = LocalLossBlockConv(ch_in=128, ch_out=128, kernel_size=3, stride=1, padding=1, dim_out=16, **params)
        self.conv5 = LocalLossBlockConv(ch_in=128, ch_out=128, kernel_size=3, stride=1, padding=1, dim_out=16, **params)


        if 'dim_decoder' in params.keys():
            params.pop('dim_decoder')

        self.fc1 = LocalLossBlockLinear(8 * 8 * 128, 1024, **params, last_layer=True)

        
    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        self.conv3.set_learning_rate(lr)
        self.conv4.set_learning_rate(lr)
        self.conv5.set_learning_rate(lr)
        self.fc1.set_learning_rate(lr)
    
        
    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        self.conv3.optim_zero_grad()
        self.conv4.optim_zero_grad()
        self.conv5.optim_zero_grad()
        self.fc1.optim_zero_grad()  

        
    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        self.conv3.optim_step()
        self.conv4.optim_step()
        self.conv5.optim_step()
        self.fc1.optim_step()

    def forward(self, x, y, y_onehot, time_window=10):
        loss_total = 0
        if self.encoding == 'latency':
            x = ((time_window-1) - (time_window-1)*x).round().long()
            input_spike = torch.zeros(time_window, *x.size(), device=y_onehot.device)
            input_spike = input_spike.scatter_(0, x.view(1,*x.size()), 1)
            x = input_spike 

        x, loss = self.conv1(x, y, y_onehot, time_window)
        loss_total += loss
        x, loss = self.conv2(x, y, y_onehot, time_window)
        loss_total += loss
        x = average_pooling_through_time(x, time_window)
        
        x, loss = self.conv3(x, y, y_onehot, time_window)
        loss_total += loss
        x, loss = self.conv4(x, y, y_onehot, time_window)
        loss_total += loss
        x, loss = self.conv5(x, y, y_onehot, time_window)
        loss_total += loss
        x = average_pooling_through_time(x, time_window)
        
        x = x.view(x.size(0), x.size(1), -1)
        x, loss = self.fc1(x, y, y_onehot, time_window)    
        loss_total += loss

        return x, loss_total
