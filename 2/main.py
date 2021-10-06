import cv2 as cv
# imread第二个参数为0表示灰度图片
import numpy as np

img = cv.imread("./shop_qrcode.jpg")
# 图片左上角为原点，输出x=200,y=300位置的bgr像素值
# 为图片手动写入像素值, 蓝色横线
# for i in range(100):
#     img[100, 100+i] = (255, 0, 0)

height, width, mode = img.shape
cv.imshow("Image", img)

# cv.imshow("chippedImage", img[100:200, 200:300])
# cv.imshow("movedImage",
#           cv.warpAffine(img, np.float32([
#               [1, 0, 100],
#               [0, 1, 200]
#           ]), (height, width))
# )

cv.waitKey(0)
cv.destroyAllWindows()



# cv.imwrite("test.jpg", img, [cv.IMWRITE_JPEG_QUALITY, 0])  # 设置图片质量，范围0-100
# cv.imwrite("test.png", img, [cv.IMWRITE_PNG_COMPRESSION, 100])  # 设置图片质量，范围0-100
# IMWRITE_JPEG_QUALITY越小，压缩比越高
# IMWRITE_PNG_COMPRESSION越小，压缩比越低
# 图片大小：宽*高*3*8 bit
# png图片，除了RGB，还有alpha通道，无损压缩
