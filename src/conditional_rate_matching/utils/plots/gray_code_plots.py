import torch
import torch as T
import torch.nn.functional as F

import tqdm
import time
import random
import sys, os
import numpy as np
from matplotlib import pyplot as plt
from sympy.combinatorics.graycode import GrayCode

def get_binmap(discrete_dim, binmode):
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append('0' + bx)
        all_bins.append('1' + bx)
    vals = all_bins[:]
    if binmode == 'rand':
        print('remapping binary repr with random permute')
        random.shuffle(vals)
    elif binmode == 'gray':
        print('remapping binary repr with gray code')
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append('0' + x)
            vals.append('1' + x)
    else:
        assert binmode == 'normal'
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm


def compress(x, discrete_dim):
    bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
    bx = '0' + bx if x >= 0 else '1' + bx
    return bx


def recover(bx):
    x = int(bx[1:], 2)
    return x if bx[0] == '0' else -x


def float2bin(samples, bm, int_scale, discrete_dim):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * int_scale
        bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float(samples, inv_bm, int_scale, discrete_dim):
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[:discrete_dim // 2], s[discrete_dim // 2:]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= int_scale
        y /= int_scale
        floats.append((x, y))
    return np.array(floats)


def plot_heat(score_func, bm, size, device, int_scale, discrete_dim, out_file=None):
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm, int_scale, discrete_dim)
    heat_samples = torch.from_numpy(heat_samples).to(device).float()
    heat_score = F.softmax(score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    print("energy max and min:", a.max(), a.min())
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    # if out_file is None:
    #     out_file = os.path.join(save_dir, 'heat.pdf')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_samples(samples, out_name, lim=None, axis=True):
    plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    plt.axis('equal')
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    if not axis:
        plt.axis('off')
    if out_name is not None:
        plt.savefig(out_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()