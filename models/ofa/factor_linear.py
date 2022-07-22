from telnetlib import X3PAD
import torch.nn as nn
import torch.nn.functional as F
import torch

class FactorLinear(nn.Module):
    def __init__(
        self,
        input_size,
        orthogonal_size,
        output_size,
        bias=False
    ):
        super(FactorLinear, self).__init__()
        # for i in range(num_modality_groups):
        self.bias = bias
        orthogonal_size = 32
        # orthogonal_size = min(input_size, output_size)
        self.add_module(
            "ms_linear",
            nn.Linear(input_size, orthogonal_size, bias=False)
        )
        # self.add_module(
        #     "original",
        #     nn.Linear(input_size, output_size)
        # )

        # for i in range(num_modality_groups):
           
        # self.add_module(
        #     "orthogonal_ms_linear",
        #     nn.utils.parametrizations.orthogonal(nn.Linear(orthogonal_size, orthogonal_size, bias=False), orthogonal_map="cayley") 
        # )
        self.add_module(
            "orthogonal_ms_linear",
            nn.Linear(orthogonal_size, orthogonal_size, bias=False)
        )
        
        self.s_linear = nn.Linear(orthogonal_size, output_size, bias=bias)


    def forward(self, x):
        x = getattr(self, "ms_linear")(x)
        # x = F.normalize(x, dim=-1)
        x = getattr(self, "orthogonal_ms_linear")(x)
        x = self.s_linear(x)
        return x
