import torch
import torch.nn as nn

#######################gg###########################################################
# Coordinate -> spectrum network (adapted from the gene-expression `hsd` model).
#   input:  (x, y) pixel coordinate        in R^2
#   output: spectrum (C spectral channels) in R^C
# Two parts:
#   (1) spatial embedding  f_S : R^2 -> R^{latent_dim}
#   (2) spectral function  f_A : R^{latent_dim} -> R^C   (was the gene head)
##################################################################################

class SpectralModel(nn.Module):
    """
    Parameters
    ----------
    C : int
        number of spectral channels (e.g. 512 wavelength bins)
    S_hidden_list : list[int]
        hidden sizes for f_S  (e.g. [64, 64])
    A_hidden_list : list[int]
        hidden sizes for f_A  (e.g. [128, 128])
    activation_fn : nn.Module
        activation used between layers
    latent_dim : int
        width of the spatial bottleneck z.
        1 reproduces the original single-"depth" model;
        >1 relaxes it so 2-D spatial structure is not forced through one scalar.
    """
    def __init__(self, C, S_hidden_list, A_hidden_list,
                 activation_fn=nn.ReLU(), latent_dim=1):
        super().__init__()

        input_size = 2  # (x, y) coordinate

        # --- spatial embedding  f_S : R^2 -> R^latent_dim ---
        S_dims = [input_size] + list(S_hidden_list) + [latent_dim]
        S_layers = []
        for l in range(len(S_dims) - 1):
            S_layers.append(nn.Linear(S_dims[l], S_dims[l + 1]))
            if l != len(S_dims) - 2:            # no activation on last layer
                S_layers.append(activation_fn)
        self.spatial_embedding = nn.Sequential(*S_layers)

        # --- spectral function  f_A : R^latent_dim -> R^C ---
        A_dims = [latent_dim] + list(A_hidden_list) + [C]
        A_layers = []
        for l in range(len(A_dims) - 1):
            A_layers.append(nn.Linear(A_dims[l], A_dims[l + 1]))
            if l != len(A_dims) - 2:            # no activation on last layer
                A_layers.append(activation_fn)
        self.spectral_function = nn.Sequential(*A_layers)

    def forward(self, x):
        z = self.spatial_embedding(x)           # latent spatial code
        return self.spectral_function(z)        # predicted spectrum (... x C)