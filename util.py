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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# single-sequence gen.
def generate_autocorr(T, rho, lag, seed=None):
    """inputs: T (number of timesteps), rho (correlation with lagged value), seed"""
    rng = np.random.default_rng(seed) 
    eps = rng.standard_normal(T)
    s = (1 - rho**2)**0.5

    r = np.empty(T)
    r[:lag] = eps[:lag] # warmup
    for t in range(lag, T):
        r[t] = rho * r[t-lag] + s * eps[t]
    return r

def make_dataset(num_sequences, T, rho, lag_range, seed=None):
    """inputs: num_sequences (int), lag_range (tuple[int])
    output: x, a tensor like (B,T) for positions 0 to T-1 
            y, a tensor like (B,T) for positions 1 to T (these are the targets)
            lags a tensor like (B,); index-aligned with sequences"""
    rng = np.random.default_rng(seed)
    seqs, lags = [], []
    for _ in range(num_sequences):
        L = int(rng.integers(lag_range[0], lag_range[1] + 1))
        seqs.append(generate_autocorr(T + 1, rho, L, rng))   # generate T+1
        lags.append(L) 
        # lags is (B,), storing lags. its indices align with the sequences the lags belong to.
    seqs = torch.tensor(np.array(seqs), dtype=torch.float32)  # (num_sequences, T+1)
    x = seqs[:, :-1]    # (num_sequences, T)  inputs:  positions 0..T-1
    y = seqs[:, 1:]     # (num_sequences, T)  targets: positions 1..T  (each is "next value")
    return x, y, torch.tensor(lags)

def make_dataset_lagset(num_sequences, T, rho, lag_set, seed=None):
    # similar setup, just lag_set is a defined set of a few lags rather than a range.
    rng = np.random.default_rng(seed)
    lag_set = np.asarray(lag_set)
    seqs, lags = [], []
    for _ in range(num_sequences):
        L = int(rng.choice(lag_set))
        seqs.append(generate_autocorr(T + 1, rho, L, rng)) 
        lags.append(L)
    seqs = torch.tensor(np.array(seqs), dtype=torch.float32)
    return seqs[:, :-1], seqs[:, 1:], torch.tensor(lags)


def correlation_by_lag(model, x_eval, lags_eval, device, start_offset=30, batch_size=512, verbose=True):
    """inputs:
    model
    x_eval: a tensor of generated sequences, like (B,T)
    lags_eval: a tensor like (B,) of lags, index-aligned with the sequences along the B dim
    start_offset: the "burn-in" period after the warmup period
    """
    model.eval()
    N = x_eval.shape[0] # saying x_eval (and subsequently x_cpu) are shape: (N,T)
    # forward in chunks, collect predictions 
    preds = []
    for i in range(0, N, batch_size): 
        # go from 0 to N (the actual number of passed-in sequences) in equal steps
        xb = x_eval[i:i+batch_size].to(device)
        pb, *_ = model(xb)
        preds.append(pb.cpu())
        del xb, pb
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pred = torch.cat(preds, dim=0)  # (N,T) Now we just concatenate all of the batches to the original amount.
    x_cpu, lags_cpu = x_eval.cpu(), lags_eval.cpu() #(N,T), (N,)
    T = x_cpu.shape[1]

    results = {}
    for L in lags_cpu.unique().tolist():
        #Note: lags_cpu (and lags_eval) come from tracking the lag of each individual sequence
        # so that object implicitly maps lags -> their respective sequencs
        # so rows is the sequence where lag = L from the preds and xs
        L = int(L)
        rows = torch.where(lags_cpu == L)[0]
        if len(rows) == 0:
            continue
        xr, pr = x_cpu.index_select(0, rows), pred.index_select(0, rows)
        # selecting rows with lag = L

        start = L + start_offset
        a = pr[:, start:].reshape(-1) # start (post warmup+burn-in) to end
        b = xr[:, start - L + 1 : T - L + 1].reshape(-1) #recall preds 1 ahead of x hence the +1 index.
        results[L] = float(np.corrcoef(a.numpy(), b.numpy())[0, 1]) 
        #correlating preds with correct lagged counterparts
        
    model.train()

    if verbose: #print stuff
        print("Per-lag correlation:")
        for L in sorted(results):
            print(f"  lag {L:2d}:  corr = {results[L]:+.3f}")
        print(f"\navg {np.mean(list(results.values())):.3f}")
    return results

def offset_profile(A, max_offset=150):
    T = A.shape[0]
    values_at_offset = [[] for _ in range(max_offset)]   
    # one list of attention activations per offset

    for query_pos in range(T):
        for d in range(0, min(query_pos + 1, max_offset)):
            # note: if query_pos = 0, and d is 1:
            # you'd think that would produce a negative key pos
            # but then we have for d in range(1,1), so the inner loop doesnt run
            key_pos = query_pos - d
            values_at_offset[d].append(A[query_pos, key_pos])

    # take the mean of each row/sublist. 
    avg_attention_at_offset = np.array([np.mean(v) for v in values_at_offset])
    # this is now [mean_at_offset_1, mean_at_offset_2... etc]
    # note: the resulting array is indexed by offset:
    # avg_attention_at_offset[d] = mean attention at distance d.

    # take the std of each row/sublist
    std_attention_at_offset = np.array([np.std(v) for v in values_at_offset])


    return avg_attention_at_offset, std_attention_at_offset # returns lists, indices are offset.