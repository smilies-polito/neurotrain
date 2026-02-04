import torch.nn as nn

from utils import average_pooling_through_time

class FashionCNN(nn.Module):
    def __init__(self, params):
        super(FashionCNN, self).__init__()
        self.learning_rule = params['learning_rule']

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

        if 'learning_rule' in params.keys():
            params.pop('learning_rule')
        if 'encoding' in params.keys():
            params.pop('encoding')

        self.conv1 = LocalLossBlockConv(ch_in=1, ch_out=32, kernel_size=5, stride=1, padding=0, dim_out=24, first_layer=True, **params)
        self.conv2 = LocalLossBlockConv(ch_in=32, ch_out=64, kernel_size=5, stride=1, padding=0, dim_out=8, **params)
        
        if 'dim_decoder' in params.keys():
            params.pop('dim_decoder')

        self.fc1 = LocalLossBlockLinear(4 * 4 * 64, 1024, **params, last_layer=True)

        
    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        self.fc1.set_learning_rate(lr)
    
        
    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        self.fc1.optim_zero_grad()  

        
    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        self.fc1.optim_step()

    def forward(self, x, y, y_onehot, time_window=10):
        loss_total = 0
        x, loss = self.conv1(x, y, y_onehot, time_window)
        loss_total += loss
        x = average_pooling_through_time(x, time_window)
        
        x, loss = self.conv2(x, y, y_onehot, time_window)
        loss_total += loss
        x = average_pooling_through_time(x, time_window)
        
        x = x.view(x.size(0), x.size(1), -1)
        x, loss = self.fc1(x, y, y_onehot, time_window)    
        loss_total += loss

        return x, loss_total
