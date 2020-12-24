from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio

from os import path, getcwd, listdir
from scipy import signal
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.cluster import KMeans

from string import Template


filePath = path.join(getcwd(), 'Segmentation', 'resource')


# 读取文件夹下imgType格式的图片
def read_directory(images=[], imgType='.jpg'):
    print('load image set')
    imStack = []
    images = images if len(images) else listdir(filePath)
    for fullflname in images:
        fname, ext = path.splitext(fullflname)
        if(ext == imgType):
            imgPath = path.join(filePath, fullflname)
            img = Image.open(imgPath)
            imStack.append(img)
    return imStack


# 展示filterbank
def displayFilterBank(F):
    numFilters = F.shape[2]
    filterSize = F.shape[0]
    gridSize = math.ceil(math.sqrt(numFilters))
    for i in range(numFilters):
        plt.subplot(gridSize, gridSize, i+1)
        plt.axis('off')
        plt.imshow(F[:, :, i])
    plt.show()


# 原始数据四周补-1
def pad_data(data, winSize):
    m, n = data.shape
    t1 = np.ones([winSize//2, n], dtype=int) * -1
    data = np.concatenate((t1, data, t1))
    m, n = data.shape
    t2 = np.ones([m, winSize//2], dtype=int) * -1
    data = np.concatenate((t2, data, t2), axis=1)
    return data


# 逐像素取大小为winSize*winSize的邻域数据
def gen_window_data(data, winSize):
    [x, y] = data.shape
    m = x-winSize//2*2
    n = y-winSize//2*2
    windows = np.zeros((m, n, winSize**2))
    for i in range(winSize//2, m):
        for j in range(winSize//2, n):
            windows[i, j, :] = data[i-winSize//2:i+winSize //
                                    2+1, j-winSize//2:j+winSize//2+1].reshape(winSize**2,)
    return windows


# 计算原始图像 至 meanFeats 的距离，生成隶属度矩阵 labelIm
def quantizeFeats(featIm, meanFeats):
    print('quantizeFeats start')
    height = featIm.shape[0]
    width = featIm.shape[1]
    [size, dim] = meanFeats.shape
    labelIm = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            minDis = -1
            for k in range(size):
                dis = 0
                for d in range(dim):
                    dis = dis + math.pow(featIm[i, j, d] - meanFeats[k, d], 2)
                    dis = math.sqrt(dis)
                    if minDis < 0 or dis < minDis:
                        labelIm[i, j] = k
                        minDis = dis
    return labelIm


# 生成纹理词典
def createTextons(imStack, bank, k):
    print('textons start')
    textonsData = []
    bankSize = bank.shape[2]  # 49*49*38

    for img in imStack:
        # responses = filter_response(img, bank)
        img = img.convert('L')
        img = np.array(img)
        [h, w] = img.shape
        # 初始化 滤波器结果集
        responses = np.zeros((h, w, bankSize))  # w * h * 38
        for r in range(bankSize):
            responses[:, :, r] = signal.convolve2d(
                img, bank[:, :, r],  mode='same')

        responses = responses.reshape((h*w, bankSize))
        # 结果集合并
        textonsData = np.concatenate(
            (textonsData, responses), axis=0) if len(textonsData) else responses
    kmeans = KMeans(n_clusters=k,  random_state=0).fit(textonsData)
    # kmeans [labels_, cluster_centers_]
    textons = kmeans.cluster_centers_  # 聚类  k * 38

    return textons


# 直方图
def extractTextonHists(origIm, bank, textons, winSize):
    print('extractTextonHists start')
    kCenter = textons.shape[1]
    # img => 灰度图 => numpy
    img = origIm.convert('L')
    img = np.array(img)
    [h, w] = img.shape
    bankSize = bank.shape[2]  # 49*49*38
    # 初始化 滤波器结果集
    responses = np.zeros((h, w, bankSize))  # w * h * 38
    for r in range(bankSize):
        responses[:, :, r] = signal.convolve2d(
            img, bank[:, :, r],  mode='same')
    responses = responses.reshape((h*w, bankSize))
    # 计算纹理响应到纹理集的距离，生成隶属度矩阵
    distances = cdist(responses, textons, metric='euclidean')
    # 获取每一维的最大值
    indexs = distances.argmax(axis=1)
    feattexton = indexs.reshape(h, w)
    # 图像边界处理
    feattexton = pad_data(feattexton, winSize)
    print(feattexton.shape)
    # 获取窗口
    windowMap = gen_window_data(feattexton, winSize)
    featIm = np.zeros((h, w, kCenter))
    # 统计纹理频率
    for i in range(windowMap.shape[0]):
        for j in range(windowMap.shape[1]):
            window = windowMap[i, j]
            frequency = Counter(window)
            for key in (frequency):  # 'Counter' object has no attribute 'shape'
                # padding 补的 -1
                if key != -1:
                    textonIndex = int(key)
                    count = frequency[key]
                    featIm[i, j, textonIndex] = count

    return featIm


# 对比基于颜色和纹理的分割结果
def compareSegmentations(origIm, bank, textons, winSize, numColorRegions, numTextureRegions):
    print('compareSegmentations start')

    img = np.array(origIm)
    [h, w, c] = img.shape
    colordata = img.reshape((h*w, c))
    colorCenter = KMeans(n_clusters=numColorRegions,  random_state=0).fit(
        colordata).cluster_centers_
    colorLabelIm = quantizeFeats(img, colorCenter)

    featIm = extractTextonHists(origIm, bank, textons, winSize)

    h = featIm.shape[0]
    w = featIm.shape[1]
    featImData = featIm.reshape((w*h, -1))
    textureCenter = KMeans(n_clusters=numTextureRegions,
                           random_state=0).fit(featImData).cluster_centers_
    textureLabelIm = quantizeFeats(featIm, textureCenter)
    return [colorLabelIm, textureLabelIm]


def main():

    # 1 加载图片
    imgSet = ['gumballs.jpg', 'snake.jpg', 'twins.jpg']
    imStack = read_directory(imgSet)
    # 2 加载 filterBank
    bankPath = path.join(filePath, 'filterBank.mat')
    filterBank = sio.loadmat(bankPath)['F']  # 49*49*38
    displayFilterBank(filterBank)
    # 3 生成纹理词典
    textons = createTextons(imStack, filterBank, 5)

    # imgName = 'coins.jpg'
    imgName = 'planets.jpg'
    imgPath = path.join(filePath, imgName)
    img = Image.open(imgPath)
    winSize, numColorRegions, numTextureRegions = 49, 10, 50

    [colorLabelIm, textureLabelIm] = compareSegmentations(
        img, filterBank, textons, winSize, numColorRegions, numTextureRegions)

    print(colorLabelIm)
    print(textureLabelIm)

    str = Template(
        'winSize=${winSize}, numColorRegions=${numColorRegions}, numTextureRegions=${numTextureRegions}')
    pltTitle = str.substitute(
        winSize=winSize, numColorRegions=numColorRegions, numTextureRegions=numTextureRegions)

    plt.figure(figsize=(24, 10), dpi=80)
    plt.figure(1)
    plt.title(pltTitle)
    plt.axis('off')
    ax1 = plt.subplot(131)
    ax1.title('origIm')
    ax1.imshow(img)
    ax2 = plt.subplot(132)
    ax2.title('ColorRegion')
    ax2.imshow(colorLabelIm)
    ax3 = plt.subplot(133)
    ax3.title('TextureRegion')
    ax3.imshow(textureLabelIm)
    plt.show()
    exit()


if __name__ == '__main__':
    main()
