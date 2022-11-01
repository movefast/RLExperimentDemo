import torch as T
import torch.nn as nn
import torch.nn.functional as functional


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
        if init is not None:
            nn.init.constant_(self.i2o.weight, init)
        else:
            nn.init.xavier_uniform_(self.i2o.weight)

    def forward(self, x):
        x = self.i2o(x)
        return x

class ConvNetwork(nn.Module):
    def __init__(self, state_dim, output_units, architecture, device):
        super().__init__()

        self.conv_body = ConvBody(device, state_dim, architecture)
        self.fc_head = self.layer_init_xavier(nn.Linear(self.conv_body.feature_dim, output_units))

        self.to(device)
        self.device = device

    def layer_init_xavier(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def forward(self, x):
        phi = self.conv_body(x)
        phi = self.fc_head(phi)
        return phi

class ConvBody(nn.Module):
    def __init__(self, device, state_dim, architecture):
        super().__init__()

        def size(size, kernel_size=3, stride=1, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        spatial_length, _, in_channels = state_dim
        num_units = None
        layers = nn.ModuleList()
        for layer_cfg in architecture['conv_layers']:
            layers.append(nn.Conv2d(layer_cfg["in"], layer_cfg["out"], layer_cfg["kernel"],
                                         layer_cfg["stride"], layer_cfg["pad"]))
            if not num_units:
                num_units = size(spatial_length, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
            else:
                num_units = size(num_units, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
        num_units = num_units ** 2 * architecture["conv_layers"][-1]["out"]

        self.feature_dim = num_units
        self.spatial_length = spatial_length
        self.in_channels = in_channels
        self.layers = layers
        self.to(device)
        self.device = device

    def forward(self, x):
        x = functional.relu(self.layers[0](self.shape_image(x)))
        for layer in self.layers[1:]:
            x = functional.relu(layer(x))
        return x.reshape(x.size(0), -1)

    def shape_image(self, x):
        return x.reshape(-1, self.spatial_length, self.spatial_length, self.in_channels).permute(0, 3, 1, 2)
