import numpy as np
import cv2


def singleScaleRetinex(img, sigma):  # 单尺度Retinex算法 参数分别图像和高斯核的标准差
    retinex = np.log10(img) - np.log10(
        cv2.GaussianBlur(img, (0, 0), sigma))  # r(x,y)= logS(x,y)-log[F(x,y)*s(x,y)] SSR模拟近似L(x,y)

    return retinex


def multiScaleRetinex(img, sigma_list):  # 多尺度加权平均的Retinex算法 参数分别图像和高斯核的标准差列表
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):  # 色彩恢复的多尺度Retinex算法 参数分别代表输入的图像、颜色强度调整系数和颜色平衡系数。CR
    img_sum = np.sum(img, axis=2, keepdims=True)  # keepdims这个参数为True，被删去的维度在结果矩阵中就被设置为一。

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):  # 颜色平衡
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):  # 颜色恢复的MSR方法MSRCR
    img = np.float64(img) + 1.0  # 保证精度

    img_retinex = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)  # G 增益 b偏差

    for i in range(img_msrcr.shape[2]):  # 归一化处理
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / (
                np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * 255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list):  # 色彩增益加权的autoMSRCR算法
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
    # 直接线性量化
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)
    # 根据原始的RGB的比例映射到每个通道
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])#在这一行，计算当前像素位置(y, x)处的图像img的所有通道的最大值，并将这个最大值赋给变量B
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])#映射比例A
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp
