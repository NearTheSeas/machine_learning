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


filePath = path.join(getcwd(), 'Segmentation', 'resource')


# 读取文件夹下所有图片
def read_directory(images=[], imgType='.jpg'):
    print('images loading...')
    imStack = []
    images = images if len(images) else listdir(filePath)
    for fullflname in images:
        fname, ext = path.splitext(fullflname)
        if(ext == imgType):
            imgPath = path.join(filePath, fullflname)
            img = Image.open(imgPath)
            imStack.append(img)
    return imStack


# 滤波器响应计算
def filter_response(img, bank):
    print('filter response...')
    bankSize = bank.shape[2]
    img = img.convert('L')
    [w, h] = img.size
    img = np.array(img).transpose()
    # 初始化 滤波器结果集
    responses = np.zeros((w, h, bankSize))  # w * h * 38
    for r in range(bankSize):
        # TODO!
        # if r % 22 == 0:
        responses[:, :, r] = signal.convolve2d(
            img, bank[:, :, r],  mode='same')
    responses = responses.reshape((w*h, bankSize))
    return responses


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
# TODO!
def gen_dataX(data, winSize):
    x, y = data.shape
    m = x-winSize//2*2
    n = y-winSize//2*2
    res = np.zeros([m*n, winSize**2])
    # print(m, n)  # 1000 667
    k = 0
    for i in range(winSize//2, m+winSize//2):
        for j in range(winSize//2, n+winSize//2):
            res[k, :] = np.reshape(
                data[i-winSize//2:i+winSize//2+1, j-winSize//2:j+winSize//2+1].T, (1, -1))
            k += 1
    # print(k) # 667000
    return res


# 计算原始图像 至 meanFeats 的距离，生成隶属度矩阵 labelIm
# distances = cdist(responses, textons, metric='euclidean') ？？？？
def quantizeFeats(featIm, meanFeats):
    print('quantizeFeats start')
    w = featIm.shape[0]
    h = featIm.shape[1]
    [size, dim] = meanFeats.shape
    labelIm = []
    for i in range(w):
        for j in range(h):
            minDis = 0
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
    print('textons start...')
    textonsData = []
    for img in imStack:
        # img => 灰度图 => numpy
        responses = filter_response(img, bank)
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
    [w, h] = origIm.size
    featIm = np.zeros((w, h, kCenter))  # 初始化
    responses = filter_response(origIm, bank)
    # 计算纹理响应到纹理集的距离，生成隶属度矩阵
    # Calculates squared distance between two sets of points.
    distances = cdist(responses, textons, metric='euclidean')
    # 获取每一维的最大值
    indexs = distances.argmax(axis=1)
    feattexton = indexs.reshape(w, h)
    # 图像边界处理
    feattexton = pad_data(feattexton, winSize)
    # 获取窗口数量
    # TODO!
    res = gen_dataX(feattexton, winSize)
    print(res.shape)
    # 统计纹理频率
    for i in range(res.shape[0]):
        # for j in range(res.shape[1]):
            window = res[i]
            frequency = Counter(window[:])
            for key in (frequency):  # 'Counter' object has no attribute 'shape'
                # padding 补的 -1
                # TODO!
                if key != -1:
                    textonIndex = int(key)
                    count = frequency[key]
                    featIm[i][textonIndex] = count
    return featIm


# 对比基于颜色和纹理的分割结果
def compareSegmentations(origIm, bank, textons, winSize, numColorRegions, numTextureRegions):
    print('compareSegmentations start')
    img = np.array(origIm)
    [h, w, c] = img.shape

    img = img.reshape((w, h, c))
    colordata = img.reshape((w*h, c))
    colorCenter = KMeans(n_clusters=numColorRegions,  random_state=0).fit(
        colordata).cluster_centers_
    colorLabelIm = quantizeFeats(img, colorCenter)

    featIm = extractTextonHists(origIm, bank, textons, winSize)
    w = featIm.shape[0]
    h = featIm.shape[1]
    featImData = featIm.reshape((w*h, -1))
    textureCenter = KMeans(n_clusters=numTextureRegions,
                           random_state=0).fit(featImData).cluster_centers_
    textureLabelIm = quantizeFeats(featIm, textureCenter)
    return [colorLabelIm, textureLabelIm]


def main():

    # 1 加载图片
    # TODO!
    imgSet = ['gumballs.jpg', 'snake.jpg', 'twins.jpg']
    # imgSet = ['gumballs.jpg']
    imStack = read_directory(imgSet)
    # 2 加载 filterBank
    bankPath = path.join(filePath, 'filterBank.mat')
    filterBank = sio.loadmat(bankPath)['F']
    # 3 生成纹理词典
    textons = createTextons(imStack, filterBank, 5)

    imgName = 'coins.jpg'
    # imgName = 'planets.jpg'
    imgPath = path.join(filePath, imgName)
    img = Image.open(imgPath)

    [colorLabelIm, textureLabelIm] = compareSegmentations(
        img, filterBank, textons, 49, 10, 50)

    print(colorLabelIm)
    print(textureLabelIm)
    print(colorLabelIm.shape)
    print(textureLabelIm.shape)


if __name__ == '__main__':
    main()
