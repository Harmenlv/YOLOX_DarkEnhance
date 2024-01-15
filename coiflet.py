#读取图片
import os
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from PIL import Image
import numpy as np
import cv2
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
set_ch()
for filename in os.listdir("F:\Datasets\VOC07+12\JPEGImages"):
        img = cv2.imread("F:\Datasets\VOC07+12\JPEGImages/"+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        coeffs2 = pywt.dwt2(img, 'coif3')
        LL, (LH, HL, HH) = coeffs2
        for i, a in enumerate([LL, LH, HL, HH]):
            #a = a[..., ::-1]
            if i == 0:
                cv2.imwrite("F:\Datasets\VOC07+12/1" + "/" + filename, a)
                print()
            if i == 1:
                cv2.imwrite("F:\Datasets\VOC07+12/2" + "/" + filename, a*255)
            if i == 2:
                cv2.imwrite("F:\Datasets\VOC07+12/3" + "/" + filename, a*255)
            if i == 3:
                cv2.imwrite("F:\Datasets\VOC07+12/4" + "/" + filename, a*255)



