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


# MODEL METHOD
def _head_patched(self, xin, params, pos, mask, A_patched=None):
    WQ, WK, WV, WO = params
    if A_patched is not None:
        return A_patched, (A_patched @ (xin @ WV)) @ WO
    Q = self.apply_rope(xin @ WQ, pos)
    K = self.apply_rope(xin @ WK, pos)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
    scores = scores.masked_fill(mask, float('-inf'))
    A = F.softmax(scores, dim=-1)
    return A, (A @ (xin @ WV)) @ WO 

# MODEL METHOD
def forward_attention_ablated(self, r, A_patch, patch_li): 
    """patches one attention matrix"""
    #can patch with the identity matrix if ablating
    B, T = r.shape
    pos = torch.arange(T, device=r.device).unsqueeze(0).expand(B, T)
    x = self.W_r(r.unsqueeze(-1))

    idx = torch.arange(T, device=r.device)
    mask = (idx.unsqueeze(0) > idx.unsqueeze(1))

    attns = [] 
    resids_post_attn = [] 
    resids_post_mlp = []
    for li, heads in enumerate(self.layers):
        delta = 0
        if li != patch_li:
            for head_params in heads:
                A, d = self._head(x, head_params, pos, mask)
                delta = delta + d
                attns.append(A)
        else: 
            for head_params in heads:
                A, d = self._head(x, head_params, pos, mask, A_patched=A_patch)
                delta = delta + d
                attns.append(A)
        x = x + delta
        resids_post_attn.append(x)        

        if self.use_mlp:
            x = x + self.mlps[li](x)
        resids_post_mlp.append(x) 

    pred = self.W_U(x).squeeze(-1)
    return pred, attns, resids_post_attn, resids_post_mlp

# MODEL METHOD
def forward_point_patch(self, r, patch_li=None, patch_point=None,
                    patch_value=None, A_patch=None, patch_hi=0):
    # NOTE: r is expected to be a tensor with shape: (B,T)
    """
    Generic patched forward. Mirrors forward() exactly when called with no patch args.

    patch_point:
        None           -> normal forward
        "attn_matrix"  -> replace attention matrix A with A_patch
        "attn_output"  -> replace summed attention delta for layer patch_li
        "post_attn"    -> replace residual stream after attention
        "mlp_output"   -> replace MLP delta for layer patch_li
        "post_mlp"     -> replace residual stream after MLP

    patch_value: a tensor of the correct shape, already constructed externally.
        Caller is responsible for matching device, dtype, and batch size.
    patch_head = the index of the head to patch
    """
    B, T = r.shape
    pos = torch.arange(T, device=r.device).unsqueeze(0).expand(B, T)

    x = self.W_r(r.unsqueeze(-1))                        # (B, T, d_model)

    idx = torch.arange(T, device=r.device)
    mask = (idx.unsqueeze(0) > idx.unsqueeze(1))         # (T, T)

    attns = []
    resids_post_attn = []
    resids_post_mlp = []
    deltas_attn = []
    deltas_mlp = []

    for li, heads in enumerate(self.layers): # 0,layer1heads --- 1, layer1heads..... etc.
        delta = 0
        for hi, head_params in enumerate(heads):
            if (patch_point == "attn_matrix" and patch_li == li and patch_hi == hi):
                WQ, WK, WV, WO = head_params
                d = (A_patch @ (x @ WV)) @ WO
                attns.append(A_patch)        # this head used A_patch
            else:
                A, d = self._head(x, head_params, pos, mask)
                attns.append(A)              # this head used its real A
            delta = delta + d

        if patch_point == "attn_output" and patch_li == li:
            delta = patch_value

        deltas_attn.append(delta)
        x = x + delta

        if patch_point == "post_attn" and patch_li == li:
            x = patch_value

        resids_post_attn.append(x)

        if self.use_mlp:
            mlp_delta = self.mlps[li](x)
            if patch_point == "mlp_output" and patch_li == li:
                mlp_delta = patch_value
            deltas_mlp.append(mlp_delta)
            x = x + mlp_delta

        if patch_point == "post_mlp" and patch_li == li:
            x = patch_value

        resids_post_mlp.append(x) 

    pred = self.W_U(x).squeeze(-1) # recall squeeze since when we unembed the d_model got converted to 1... a 3rd dimension.
    return pred, attns, resids_post_attn, resids_post_mlp, deltas_attn, deltas_mlp






# MODEL METHOD
def repeated_ablation(model, x, patch_li, A_patch=None, mode="identity",num_repeats=1, return_cache=True):
    """model            trained AutocorrRoPE model
        x                input batch, shape (B, T)
        patch_li         layer to ablate / patch repeatedly
        num_repeats      how many times to re-apply that patched attention block
        A_patch          optional custom attention matrix
        patch_mode       "identity", "zero", "stripe", or "custom"
        stop_after_layer optional cutoff if you only want partial forward
        return_cache     whether to return attns/resids too"""