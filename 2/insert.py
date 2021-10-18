import cv2 as cv
import numpy as np
import math



# 图像的缩放
# 最近邻域插值法：srcX=dstX*(srcWidth/dstWidth)，srcY=dstY*(srcHeight/dstHeight)
# srcX：原图的x点处的像素，dstX：目标图片的x点处的像素
# 原图的像素位置对应目标图片的像素位置，若计算结果是小数，取最近的一个点
def nearest_insert(img, scale):
    height, width, mode = img.shape
    dstWidth = int(width * scale)
    dstHeight = int(height * scale)
    dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
    for dstX in range(dstHeight):
        for dstY in range(dstWidth):
            srcX = round(dstX * (height / dstHeight))
            srcY = round(dstY * (width / dstWidth))
            srcX = srcX - 1 if srcX > 255 else srcX
            srcY = srcY - 1 if srcY > 255 else srcY
            dstImage[dstX, dstY] = img[srcX, srcY]

    # for srcX in range(height):
    #     for srcY in range(width):
    #         dstX = int(srcX * (dstHeight/height))
    #         dstY = int(srcY * (dstWidth/width))
    #         dstImage[dstX, dstY] = img[srcX, srcY]

    return dstImage


def double_liner_insert(img, scale):
    height, width, mode = img.shape
    dstWidth = int(width * scale)
    dstHeight = int(height * scale)
    dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
    for dstX in range(dstHeight):
        for dstY in range(dstWidth):
            srcX = dstX * (height / dstHeight)
            srcY = dstY * (width / dstWidth)
            srcX = srcX - 1 if srcX > 255 else srcX
            srcY = srcY - 1 if srcY > 255 else srcY
            x = math.floor(srcX)
            y = math.floor(srcY)
            u = srcX - x  # 权值
            v = srcY - y
            dstImage[dstX, dstY] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[
                x, y + 1] + u * v * img[x + 1, y + 1]

    return dstImage


def bi_bubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/2/shop_qrcode.jpg")
    cv.imshow("Image", img)
    n = 5
    cv.imshow("nearest dstImage", nearest_insert(img, n))
    cv.imshow("double liner dstImage", double_liner_insert(img, n))
    print("success")
    cv.waitKey(0)
    cv.destroyAllWindows()
