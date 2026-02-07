import math

import torch.nn as nn
import torch


class LinearFAFunction(torch.autograd.Function):
    '''
    Autograd function for linear KP module .
    '''
    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight, grad_bias

class LinearFA(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        if torch.cuda.is_available():
            self.weight.data = self.weight.data.cuda()
            self.weight_fa.data = self.weight_fa.data.cuda()
            if bias:
                self.bias.data = self.bias.data.cuda()
    
    def reset_parameters(self):
        # stdv = 1.0 / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.weight_fa.data.uniform_(-stdv, stdv)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()
            
    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ', bias=' + str(self.bias is not None) + ')'
