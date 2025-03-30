import torch
import torch.nn as nn

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        res = self.residual(x) if self.residual is not None else x
        return out + res

class KoopmanNet(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(KoopmanNet, self).__init__()

        layers_list = []
        for layer_i in range(len(encode_layers) - 1):
            layers_list.append(ResidualBlock(encode_layers[layer_i], encode_layers[layer_i + 1]))

        self.encode_net = nn.Sequential(*layers_list)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim

        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        with torch.no_grad():
            U, _, V = torch.svd(self.lA.weight.data)
            self.lA.weight.data = torch.mm(U, V.t()) * 0.9

        if u_dim is not None:
            self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)

    def forward(self, x, b):
        return self.lA(x) + self.lB(b) if self.u_dim is not None else self.lA(x)