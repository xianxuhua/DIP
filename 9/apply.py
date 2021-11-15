import cv2 as cv
import numpy as np
from skimage.morphology import erosion, dilation, skeletonize, convex_hull_image, convex_hull_object
from scipy.ndimage.morphology import binary_hit_or_miss, binary_fill_holes


def extract_connect_component(img):
    count = 0
    core = np.ones([3, 3])
    X0 = np.zeros(img.shape, np.uint8)
    while img.any():
        # 在原图上查找像素不为0的点
        xs, ys = np.where(img > 0)
        # 把第一个点赋给新图像X0
        X0[xs[0], ys[0]] = 255
        while 1:
            X1 = cv.bitwise_and(img, dilation(X0, core))
            # 找到一个连通分量
            if (X0 == X1).all():
                count += 1
                # 每提取到一个连通分量，原图减去该分量
                img -= X1
                break
            else:
                X0 = X1

    print("连通分量", count)
    cv.imshow("img left", img)


def refine(img):
    core = np.ones([3, 3])
    return img - binary_hit_or_miss(img / 255, core)


def coarsening(img):
    return np.bitwise_or(img.astype(np.byte), binary_hit_or_miss(img, np.ones([30, 30])).astype(np.byte))


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/9/balls-with-reflections.tif", 0)
    # img = cv.imread("/Users/xxh/projects/python/ml/9/chickenfilet-with-bones.tif", 0)
    _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    cv.imshow("img", img)
    # extract_connect_component(img)
    # cv.imshow("refine", refine(img))
    # cv.imshow("fill hole", binary_fill_holes(img).astype(np.uint8) * 255)
    # cv.imshow("skeletonize", skeletonize(img / 255).astype(np.uint8) * 255)
    # 凸壳，生成一个凸多边形，把所有值为1的点包含在内
    # cv.imshow("convex_hull_object", convex_hull_object(img / 255).astype(np.uint8) * 255)
    # 粗化
    img = refine(img)
    cv.imshow("res", coarsening(img / 255).astype(np.uint8) * 255)
    cv.waitKey(0)
    cv.destroyAllWindows()
