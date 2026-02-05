import torch.nn as nn

class MNISTDNN(nn.Module):
    def __init__(self, params):
        super(MNISTDNN, self).__init__()
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

        if 'dim_decoder' in params.keys():
            params.pop('dim_decoder')

        if 'encoding' in params.keys():
            params.pop('encoding')

        self.fc1 = LocalLossBlockLinear(784, 800, first_layer=True, **params, last_layer=True)

    def set_learning_rate(self, lr):
        self.fc1.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.fc1.optim_zero_grad()
       
    def optim_step(self):
        self.fc1.optim_step()
            
    def forward(self, x, y, y_onehot, time_window=10):
        total_loss = 0.0
        x = x.view(x.size(0), -1)
        x, loss = self.fc1(x, y, y_onehot, time_window)
        total_loss += loss        
        return x, total_loss