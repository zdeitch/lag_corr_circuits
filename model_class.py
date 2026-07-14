
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

class AutocorrRoPE(nn.Module):
    def __init__(self, d_model=64, d_head=64, n_layers=2, n_heads=1, use_mlp=False, act_function='GELU'): # defaults
        super().__init__()
        self.use_mlp = use_mlp
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_r = nn.Linear(1, d_model)


        k = torch.arange(0, d_head // 2).float()
        self.register_buffer('theta', 1.0 / (10000 ** (2 * k / d_head)))

        def head_params(): # initializes each of the head weights (for one head) 
            return nn.ParameterList([
                nn.Parameter(torch.randn(d_model, d_head) * 0.02),   # WQ 
                nn.Parameter(torch.randn(d_model, d_head) * 0.02),   # WK
                nn.Parameter(torch.randn(d_model, d_head) * 0.02),   # WV
                nn.Parameter(torch.randn(d_head, d_model) * 0.02),   # WO
            ])


        # Ok so layers[L] is a list of heads; each head is a ParameterList [WQ,WK,WV,WO]
        """[nn.Parameterlist(head_params()) for _ in range(n_heads)]
            -> This creates a list of nn.ParameterList Objects (each a "head" as stated above)
                So it registers a list of heads, which themselves are lists of weights.

                And for _ in range(n_heads) initializes one of these [WQ,WK,WV,WO] lists per layer.

                Finally, the outer for _ in range(n_layers) controls the number of distinct 
                sets of [WQ,WK,WV,WO]
                (i.e. layers)
        """

        self.layers = nn.ModuleList([
            nn.ModuleList([nn.ParameterList(head_params()) for _ in range(n_heads)])
            for _ in range(n_layers)
        ])


        """If MLP is used, initialize and register one per layer."""
        if use_mlp:
            def make_mlp():
                act = nn.GELU() if act_function == 'GELU' else nn.ReLU()
                return nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    act,
                    nn.Linear(4 * d_model, d_model)
                )
            self.mlps = nn.ModuleList([make_mlp() for _ in range(n_layers)])

        self.W_U = nn.Linear(d_model, 1, bias=True)

    def apply_rope(self, x, pos):
        half = x.shape[-1] // 2
        angles = pos.float().unsqueeze(-1) * self.theta
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)

    def _head(self, xin, params, pos, mask):
        """takes as input:
        1) current residual stream vector
        2) the weight matrices of a given head (registered in __init__ as parameterlist)
        3) the positions"""
        WQ, WK, WV, WO = params
        Q = self.apply_rope(xin @ WQ, pos)
        K = self.apply_rope(xin @ WK, pos)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(mask, float('-inf')) # causal masking
        A = F.softmax(scores, dim=-1) 
        return A, (A @ (xin @ WV)) @ WO # the delta thats gonna be added to resid. stm.

    def forward(self, r): # r: (B, T) 
        # r is np.array of B sequences, with T values for each sequence 
        B, T = r.shape
        pos = torch.arange(T, device=r.device).unsqueeze(0).expand(B, T) # position values for sequencew (B,T)
        # just to align indices of values in sequence with positions, for use in apply_rope()

        x = self.W_r(r.unsqueeze(-1))                        # (B, T, d_model)

        idx = torch.arange(T, device=r.device)
        mask = (idx.unsqueeze(0) > idx.unsqueeze(1))         # (T, T)

        attns = [] # each ( B,T,T). I am just gonna store a list to be returned for later inspection
        resids_post_attn = [] # each (B,T,64). Storing for same reason.
        resids_post_mlp = [] 

        for li, heads in enumerate(self.layers): # li -> "layer index"
            """looping over the layers.
            Recall each layer is list of heads, which themselves are lists of their four weight matrices.
            So it goes:
                layer 0 -> heads in layer 0
                layer 1 -> heads in layer 1
                etc
            """
            delta = 0 # reset delta, so heads from previous layers don't add theirs again
            for head_params in heads:
                """now we actually access the heads.
                heads is a parameterList like [head1, head2... etc] and each head is its own 
                parameterList of randomly initialized Weight matrices"""

                A, d = self._head(x, head_params, pos, mask)
                delta = delta + d # each head adds their delta residual stm. vec
                attns.append(A)
            x = x + delta # update the actual resid. stream
            resids_post_attn.append(x) # tracks each layer's residual stream, but before MLP

            if self.use_mlp:
                x = x + self.mlps[li](x) # this line suffices since each mlp is a nn.Sequential
            resids_post_mlp.append(x) # tracks residual stream AFTER MLP
            

        pred = self.W_U(x).squeeze(-1)  # (B, T)
        return pred, attns, resids_post_attn, resids_post_mlp