import math
import os
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  # Specify the default font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the issue of negative sign '-' displaying as a square when saving images
from PIL import Image
import numpy as np

def MinRgb(c):
    return min(c[0], c[1], c[2])

def SumRgb(c):
    return c[0] + c[1] + c[2]

def Invert(img):
    img = 255 - img
    return img

def GetA(R, G, B, k=100):
    # k is set to 100 by default to get the top 100 pixels after sorting in the original text
    rlist = []
    height, width = R.shape[0], R.shape[1]
    for hi in range(height):
        for wi in range(width):
            rlist.append([R[hi][wi], G[hi][wi], B[hi][wi]])

    rlist.sort(key=MinRgb)
    rlist.reverse()
    rlist = rlist[:k]
    rlist.sort(key=SumRgb)
    rlist.reverse()
    return rlist[0][0], rlist[0][1], rlist[0][2]

def CalT(R, G, B, r_A, g_A, b_A, size=1, w=0.75):
    ts = (size - 1) // 2
    height, width = R.shape[0], R.shape[1]
    R_f = np.pad(R, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / r_A
    G_f = np.pad(G, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / g_A
    B_f = np.pad(B, ((ts, ts), (ts, ts)), 'constant', constant_values=(255, 255)) / b_A

    shape = (height, width, size, size)
    strides = R_f.itemsize * np.array([width + ts * 2, 1, width + ts * 2, 1])

    blocks_R = np.lib.stride_tricks.as_strided(R_f, shape=shape, strides=strides)
    blocks_G = np.lib.stride_tricks.as_strided(G_f, shape=shape, strides=strides)
    blocks_B = np.lib.stride_tricks.as_strided(B_f, shape=shape, strides=strides)
    l = []
    i = 0
    t = np.zeros((height, width))
    for hi in range(height):
        for wi in range(width):
            t[hi, wi] = 1 - w * min(np.min(blocks_R[hi, wi]), np.min(blocks_G[hi, wi]), np.min(blocks_B[hi, wi]))
            if t[hi, wi] < 0.5:
                i = i + 1
                t[hi, wi] = 2 * t[hi, wi] * t[hi, wi]
                if i % 10 == 0:
                    l.append(t[hi, wi] * 100)
    plt.figure(figsize=(100, 20), dpi=300)
    plt.scatter(range(0, len(l)), l, c='red', label='t(x)')
    plt.xticks([])
    plt.yticks(range(10, 50, 5), fontsize=150)
    plt.xlabel("b", fontdict={'size': 150})
    plt.ylabel("t(x)", fontdict={'size': 150})
    plt.savefig("figures.jpg", dpi=600)
    return t

def DeHaze(filepath, filename):
    img = Image.open(filepath)
    img = np.asarray(img, dtype=np.int32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R, G, B = Invert(R), Invert(G), Invert(B)
    r_A, g_A, b_A = GetA(R, G, B)
    t = CalT(R, G, B, r_A, g_A, b_A)
    J_R = (R - r_A) / t + r_A
    J_G = (G - g_A) / t + g_A
    J_B = (B - b_A) / t + b_A
    J_R, J_G, J_B = Invert(J_R), Invert(J_G), Invert(J_B)
    r = Image.fromarray(J_R).convert('L')
    g = Image.fromarray(J_G).convert('L')
    b = Image.fromarray(J_B).convert('L')
    image = Image.merge("RGB", (r, g, b))

    image.save("f:/" + filename)

if __name__ == '__main__':
    DeHaze(r"F:\0.2.jpg", "q1.jpg")
