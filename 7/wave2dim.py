import pywt
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet, estimate_sigma


# 低频分量，水平高频、垂直高频、对角线高频
# cA, (cH, cV, cD) = pywt.dwt2(img, 'db1')
# print(img.shape)
# print(cA.shape)
# print(cD.shape)
