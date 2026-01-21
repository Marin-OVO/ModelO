import torch
import torch.nn as nn
import math


def dct_1d(x):
    orig_dtype = x.dtype
    x = x.to(torch.float32)

    N = x.shape[-1]
    v = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    W = torch.exp(-1j * math.pi * k / (2 * N))

    X = (V[..., :N] * W).real
    X[..., 0] = X[..., 0] / math.sqrt(N)
    X[..., 1:] = X[..., 1:] / math.sqrt(N / 2.0)

    return X.to(orig_dtype)

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

def build_frequency_mask(H, W, low_ratio, high_ratio, device):
    u = torch.arange(H, device=device).float()
    v = torch.arange(W, device=device).float()

    uu, vv = torch.meshgrid(u / H, v / W, indexing='ij')
    freq_dist = torch.sqrt(uu ** 2 + vv ** 2)

    freq_dist = freq_dist / freq_dist.max()

    mask = torch.zeros_like(freq_dist)
    mask[(freq_dist >= low_ratio) & (freq_dist < high_ratio)] = 1.0

    sigma = 0.05

    low_transition = torch.exp(-((freq_dist - low_ratio) ** 2) / (2 * sigma ** 2))
    high_transition = torch.exp(-((freq_dist - high_ratio) ** 2) / (2 * sigma ** 2))

    mask = torch.where(
        freq_dist < low_ratio,
        low_transition,
        torch.where(freq_dist >= high_ratio, high_transition, mask)
    )

    return mask