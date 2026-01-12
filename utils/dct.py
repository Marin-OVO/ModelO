import torch
import torch.nn as nn
import math


def dct_1d(x):
    N = x.shape[-1]
    v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device).float()
    W = torch.exp(-1j * math.pi * k / (2 * N))
    X = (V[..., :N] * W).real
    X[..., 0] /= math.sqrt(N)
    X[..., 1:] /= math.sqrt(N / 2)
    return X

def idct_1d(X):
    N = X.shape[-1]
    X = X.clone()
    X[..., 0] *= math.sqrt(N)
    X[..., 1:] *= math.sqrt(N / 2)
    k = torch.arange(N, device=X.device).float()
    W = torch.exp(1j * math.pi * k / (2 * N))
    V = torch.zeros(X.shape[:-1] + (2 * N,),
                    device=X.device,
                    dtype=torch.complex64)
    V[..., :N] = X * W
    v = torch.fft.ifft(V, dim=-1).real
    return v[..., :N]

def dct_2d(x):
    return dct_1d(dct_1d(x).transpose(-1, -2)).transpose(-1, -2)

def idct_2d(x):
    return idct_1d(idct_1d(x).transpose(-1, -2)).transpose(-1, -2)

def build_hpf_mask(h, w, alpha, device):
    mask = torch.ones((h, w), device=device)
    h_cut = int(h * alpha)
    w_cut = int(w * alpha)
    mask[:h_cut, :w_cut] = 0.0

    return mask