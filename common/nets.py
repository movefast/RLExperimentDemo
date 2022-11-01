import math

import torch as T
import torch.nn as nn


def lin_block(input_size, output_size, actn, bias, dropout=False):
    layers = [nn.Linear(input_size, output_size, bias), actn()]
    if dropout:
        layers.insert(1, nn.Dropout(p=0.5))
    return nn.Sequential(*layers)


def weight_init(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class FCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, actn, bias):
        super().__init__()
        dims = (input_size,) + tuple(hidden_sizes)
        self.layers = nn.Sequential(*[lin_block(dim_in, dim_out, actn, bias) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.h2o = nn.Linear(dims[-1], output_size, bias=bias)
        self.apply(weight_init)

    def forward(self, x):
        x = self.layers(x)
        x = self.h2o(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, bias):
        super(SimpleNN, self).__init__()
        self.nlin = nn.ReLU()
        # 2-layer nn
        self.i2h = nn.Linear(input_size, input_size//2, bias=bias)
        self.h2o = nn.Linear(input_size//2, output_size, bias=bias)

    def forward(self, x):
        # 2-layer nn
        x = self.i2h(x)
        x = self.nlin(x)
        x = self.h2o(x)
        return x

class LinearNN(nn.Module):
    def __init__(self, input_size, output_size, init=None):
        super(LinearNN, self).__init__()
        self.i2o = nn.Linear(input_size, output_size, bias=False)
        # nn.init.constant_(self.i2o.weight, 1.0/input_size)
        # nn.init.normal_(self.i2o.weight, 0, 1/math.sqrt(input_size))
        # nn.init.normal_(self.i2o.weight, 0, math.sqrt(2/(input_size + output_size)))
        # nn.init.xavier_normal_(self.i2o.weight)
        # nn.init.xavier_uniform_(self.i2o.weight)
        if init is not None:
            nn.init.constant_(self.i2o.weight, init)
        else:
            nn.init.xavier_uniform_(self.i2o.weight)
        # nn.init.constant_(self.i2o.weight, 1.0/16)
        # nn.init.constant_(self.i2o.weight, 0)

    def forward(self, x):
        x = self.i2o(x)
        return x
