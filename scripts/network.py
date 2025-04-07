import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

        self.residual = (
            nn.Linear(in_features, out_features) if in_features != out_features else None
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        if self.residual is not None:
            nn.init.kaiming_uniform_(self.residual.weight, nonlinearity='relu')
            if self.residual.bias is not None:
                nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        res = self.residual(x) if self.residual is not None else x
        return out + res


class KoopmanNet(nn.Module):
    def __init__(self, encode_layers, Nkoopman: int, u_dim: int | None):
        super().__init__()

        self.encode_net = nn.Sequential(
            *[
                ResidualBlock(encode_layers[i], encode_layers[i + 1])
                for i in range(len(encode_layers) - 1)
            ]
        )

        self.Nkoopman = Nkoopman
        self.u_dim = u_dim

        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        if u_dim is not None:
            self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.lA.weight, nonlinearity='relu')
        if hasattr(self, "lB"):
            nn.init.kaiming_uniform_(self.lB.weight, nonlinearity='relu')

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, b=None):
        koop = self.lA(x)
        if self.u_dim is not None and b is not None:
            koop = koop + self.lB(b)
        return koop