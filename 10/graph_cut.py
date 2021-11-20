import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("/Users/xxh/projects/python/ml/10/jxrg.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# 创建与图像大小相同的黑色掩膜
mask = np.zeros(img.shape[:2], np.uint8)
# 定义前景和背景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 预定义包含想分割出来前景的矩形
rect = (50, 20, 210, 270)

# 调用GrabCut方法
# 第6个参数为迭代次数
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
# 调用GrabCut后掩膜mask只有0~3的值，0-背景，1-前景，2-可能的背景，3-可能的前景
# 使用no.where方法将0和2转为0，1和3转为1，然后保存在mask2中
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 使用mask2将背景与前景部分区分开来
res = img * mask2[:, :, np.newaxis]
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.subplot(2, 1, 2)
plt.imshow(res)
plt.show()

