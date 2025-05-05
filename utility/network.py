import torch
import torch.nn as nn

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.residual = (
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        )

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        res = self.residual(x)
        return out + res

class FeedforwardBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class KoopmanNet(nn.Module):
    def __init__(self, encode_layers, Nkoopman: int, u_dim: int | None, use_residual: bool = True):
        super().__init__()

        Block = ResidualBlock if use_residual else FeedforwardBlock

        self.encode_net = nn.Sequential(
            *[
                Block(encode_layers[i], encode_layers[i + 1])
                for i in range(len(encode_layers) - 1)
            ]
        )

        self.Nkoopman = Nkoopman
        self.u_dim = u_dim

        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        if u_dim is not None:
            self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9

        if isinstance(self.encode_net[-1], ResidualBlock):
            nn.init.orthogonal_(self.encode_net[-1].linear.weight)
        elif isinstance(self.encode_net[-1], FeedforwardBlock):
            nn.init.orthogonal_(self.encode_net[-1].block[0].weight)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, b=None):
        koop = self.lA(x)
        if b is not None:
            koop = koop + self.lB(b)
        return koop
