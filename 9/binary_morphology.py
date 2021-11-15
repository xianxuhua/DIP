import numpy as np
from skimage.morphology import erosion, dilation, opening, closing
from scipy.ndimage.morphology import binary_hit_or_miss
import cv2 as cv


if __name__ == '__main__':
    img = cv.imread("/Users/xxh/projects/python/ml/9/wingding-square-solid.tif", 0)
    cv.imshow("img", img)
    # 腐蚀
    res = erosion(img, np.ones([3, 3]))
    # 膨胀
    # res = dilation(img, np.ones([3, 3]))
    cv.imshow("res", res)
    cv.imshow("diff", img-res)

    # hmt
    img = np.ones([1000, 1000])
    img[500:, 500:] = 0
    cv.imshow("img", (img * 255).astype(np.uint8))
    res: np.ndarray = binary_hit_or_miss(img / 255, np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
    ]))
    cv.imshow("res", (res * 255).astype(np.uint8))
    cv.imshow("img", img)

    core = np.ones([3, 3])
    # open close
    cv.imshow("res", closing(opening(img, core), core))

    cv.waitKey(0)
    cv.destroyAllWindows()
