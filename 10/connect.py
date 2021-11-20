import cv2 as cv
import numpy as np


def bounds225(res):
    return np.uint8(255 * normalization(res))


def normalization(res):
    mi, ma = np.min(res), np.max(res)
    res -= mi
    return res / (ma - mi)


def gauss_core(size, K, sigma):
    core = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            core[i, j] = K * np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    core /= np.sum(core)
    return core


def relevance(matrix, core, fill_mode):
    assert len(core) % 2 == 1, 'core大小必须为奇数'
    m, n = core.shape
    height, width = matrix.shape
    res = np.zeros((height + m - 1, width + n - 1))
    m_pad = (m - 1) // 2
    n_pad = (n - 1) // 2

    pad_matrix = cv.copyMakeBorder(matrix, m_pad, m_pad, n_pad, n_pad, fill_mode)
    new_height, new_width = pad_matrix.shape

    for i in range(m_pad, new_height - m_pad):
        for j in range(n_pad, new_width - n_pad):
            res[i, j] = np.sum(core * pad_matrix[i - m_pad:i + m_pad + 1, j - n_pad:j + n_pad + 1])

    return res


def convolution(matrix, core, fill_mode=cv.BORDER_DEFAULT):
    return relevance(matrix, np.fliplr(np.flipud(core)), fill_mode)


def sub_area_connect():
    path = "/Users/xxh/projects/python/ml/10/1863694-20191206105727406-2050307115.jpg"
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray / 255.0  # 像素值0-1之间

    # sobel算子分别求出gx，gy
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=1)  # 得到梯度幅度和梯度角度阵列
    g = np.zeros(gray.shape)  # g与图片大小相同

    # 行扫描，间隔k时，进行填充，填充值为1
    def edge_connection(img, size, k):
        for i in range(size):
            Yi = np.where(img[i, :] > 0)
            if len(Yi[0]) >= 10:  # 可调整
                for j in range(0, len(Yi[0]) - 1):
                    if Yi[0][j + 1] - Yi[0][j] <= k:
                        img[i, Yi[0][j]:Yi[0][j + 1]] = 1
        return img

    # 选取边缘，提取边缘坐标，将g中相应坐标像素值设为1
    X, Y = np.where((mag > np.max(mag) * 0.3) & (ang >= 0) & (ang <= 90))
    g[X, Y] = 1

    # 边缘连接，此过程只涉及水平，垂直边缘连接，不同角度边缘只需旋转相应角度即可
    g = edge_connection(g, gray.shape[0], k=20)
    g = cv.rotate(g, 0)
    g = edge_connection(g, gray.shape[1], k=20)
    g = cv.rotate(g, 2)

    cv.imshow("img", img)
    cv.imshow("g", g)


def hough_connect():
    img = cv.imread("/Users/xxh/projects/python/ml/10/airport.tif", 0)
    img = cv.GaussianBlur(img, [5, 5], 2)
    cv.imshow("gauss", img)
    # 标准霍夫变换
    edges = cv.Canny(img, 50, 150)
    cv.imshow("canny", edges)
    # rho: 半径的搜索步长
    # theta：角度搜索步长，单位为弧度
    # threshold：只有属于同一直线的点数超过该阈值才会被检测为直线
    # 返回的就是极坐标系中的两个参数，rho和theta
    # max_theta ：检测直线的最大角度
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=250)
    lines = lines[:, 0, :]  # 将数据转换到二维
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 由参数空间向实际坐标点转换
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv.imshow('img', img)


if __name__ == '__main__':
    hough_connect()
    cv.waitKey(0)
    cv.destroyAllWindows()
