# import torch
# import torch.nn as nn

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.relu = nn.ReLU()

#         self.residual = (
#             nn.Linear(in_features, out_features) if in_features != out_features else None
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
#         if self.linear.bias is not None:
#             nn.init.zeros_(self.linear.bias)

#         if self.residual is not None:
#             nn.init.kaiming_uniform_(self.residual.weight, nonlinearity='relu')
#             if self.residual.bias is not None:
#                 nn.init.zeros_(self.residual.bias)

#     def forward(self, x):
#         out = self.linear(x)
#         out = self.relu(out)
#         res = self.residual(x) if self.residual is not None else x
#         return out + res


# class KoopmanNet(nn.Module):
#     def __init__(self, encode_layers, Nkoopman: int, u_dim: int | None):
#         super().__init__()

#         self.encode_net = nn.Sequential(
#             *[
#                 ResidualBlock(encode_layers[i], encode_layers[i + 1])
#                 for i in range(len(encode_layers) - 1)
#             ]
#         )

#         self.Nkoopman = Nkoopman
#         self.u_dim = u_dim

#         self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
#         if u_dim is not None:
#             self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.kaiming_uniform_(self.lA.weight, nonlinearity='relu')
#         if hasattr(self, "lB"):
#             nn.init.kaiming_uniform_(self.lB.weight, nonlinearity='relu')

#     def encode(self, x):
#         return torch.cat([x, self.encode_net(x)], dim=-1)

#     def forward(self, x, b=None):
#         koop = self.lA(x)
#         if self.u_dim is not None and b is not None:
#             koop = koop + self.lB(b)
#         return koop

# import math
# import torch
# import torch.nn as nn


# def gaussian_init_(n_units, std=1):
#     """Kept for backward‑compatibility (no longer used by default)."""
#     sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
#     Omega = sampler.sample((n_units, n_units))[..., 0]
#     return Omega


# class ResidualBlock(nn.Module):
#     """A simple residual MLP block with orthogonal weight initialisation."""

#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.relu = nn.ReLU()

#         # Optional skip projection if the feature dimensions change
#         self.residual = (
#             nn.Linear(in_features, out_features) if in_features != out_features else None
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Orthogonal initialisation for the encoder weights
#         nn.init.orthogonal_(self.linear.weight)
#         if self.linear.bias is not None:
#             nn.init.zeros_(self.linear.bias)

#         if self.residual is not None:
#             nn.init.orthogonal_(self.residual.weight)
#             if self.residual.bias is not None:
#                 nn.init.zeros_(self.residual.bias)

#     def forward(self, x):
#         out = self.relu(self.linear(x))
#         res = self.residual(x) if self.residual is not None else x
#         return out + res


# class KoopmanNet(nn.Module):
#     """Deep Koopman operator network with tailored weight initialisation.

#     * Encoder network (encode_net) → **Orthogonal** initialisation to encourage
#       an orthogonal set of basis functions.
#     * Linear Koopman operator (lA) and control matrix (lB) → **Kaiming**
#       initialisation (fan‑in, uniform) which is a sensible default for the rest
#       of the model.
#     """

#     def __init__(self, encode_layers, Nkoopman: int, u_dim: int | None):
#         super().__init__()

#         # Build encoder as a stack of residual blocks
#         self.encode_net = nn.Sequential(
#             *[
#                 ResidualBlock(encode_layers[i], encode_layers[i + 1])
#                 for i in range(len(encode_layers) - 1)
#             ]
#         )

#         self.Nkoopman = Nkoopman
#         self.u_dim = u_dim

#         # Koopman operator (no bias) and optional control matrix
#         self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
#         if u_dim is not None:
#             self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

#         self._reset_parameters()

#     # ---------------------------------------------------------------------
#     # Initialisation helpers
#     # ---------------------------------------------------------------------
#     def _reset_parameters(self):
#         # Encoder weights are already orthogonally initialised inside
#         # ResidualBlock; here we only take care of the remaining matrices.
#         nn.init.kaiming_uniform_(self.lA.weight, a=math.sqrt(5))
#         if hasattr(self, "lB"):
#             nn.init.kaiming_uniform_(self.lB.weight, a=math.sqrt(5))

#     # ---------------------------------------------------------------------
#     # Forward pass helpers
#     # ---------------------------------------------------------------------
#     def encode(self, x):
#         """Return concatenation of original state and learned observables."""
#         return torch.cat([x, self.encode_net(x)], dim=-1)

#     def forward(self, x, b=None):
#         """Predict next‑step Koopman latent dynamics."""
#         koop = self.lA(x)
#         if self.u_dim is not None and b is not None:
#             koop = koop + self.lB(b)
#         return koop

import math
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.residual = (
            nn.Linear(in_features, out_features) if in_features != out_features else None
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        if self.residual is not None:
            nn.init.orthogonal_(self.residual.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
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
        nn.init.orthogonal_(self.lA.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        if hasattr(self, "lB"):
            nn.init.orthogonal_(self.lB.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, b=None):
        koop = self.lA(x)
        if self.u_dim is not None and b is not None:
            koop = koop + self.lB(b)
        return koop
