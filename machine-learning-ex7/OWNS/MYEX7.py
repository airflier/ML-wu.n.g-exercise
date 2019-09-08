                #======K-means Clustering==========#

#=========================1.1Implementing K-means

#=======1.1.1Finding closet centroids

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def findClosestCentroids(X, centroids):
    #output a one-dimensional array idx that holds the index of the closest
    #centroid to every example
    idx = []
    #限制最大距离
    max_dist = 10000000
    for i in range(len(X)):
        #利用了numpy的广播属性，低维度－高纬度，（1,2）- （3,2） = （3,2）
        minus = X[i] - centroids
        dist = minus[:, 0 ]**2 + minus[:,1]**2
        if dist.min() < max_dist:
            #利用argmin找最小坐标
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)

mat = loadmat("E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex7\ex7\ex7data2.mat")
X = mat['X']   #(300, 2)
#print(X.shape)
init_centroids = np.array([[3,3], [6,2], [8,5]])
idx = findClosestCentroids(X, init_centroids)
#print(idx.shape)
#print(len([init_centroids][0])) == print(len(init_centroids))

#===============1.1.2 Computing centroid means

#重新计算centroid
def computeCentroids(X, idx):
	centroids = []
    #np.unique() means K (K 个 centroid)
	for i in range(len(np.unique(idx))):
	    u_k = X[idx == i].mean(axis = 0)#列平均值
	    centroids.append(u_k)
	return np.array(centroids)

#print(computeCentroids(X, idx))
'''
[[2.42830111 3.15792418]
 [5.81350331 2.63365645]
 [7.11938687 3.6166844 ]]
 '''

#=============1.2 K-means on example dataset

def plotData(X, centroids, idx = None):
	'''
	可视化，自动分开上色，idx：最后一次生成的idx向量，每个样本分配的中心点的值
	centroids：每次中心点的历史记录
	'''
	colors = ['b','g','gold','darkorange','salmon','olivedrab', 
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
             'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']
	
	assert len(centroids[0]) <= len(colors), 'colors not enough'

	subX = []   #分好类的样本点
	#根据最后一次的idx 对样本进行分类 并纳入subX
	if idx is not None:
		for i in range(centroids[0].shape[0]):
			x_i = X[idx == i]
			subX.append(x_i)
	else:
		subX = [X]

	plt.figure(figsize=(8, 5))
	for i in range(len(subX)):
		xx = subX[i]
		plt.scatter(xx[:,0], xx[:,1], c=colors[i], label = 'Cluster %d'%i)
	plt.legend()
	plt.grid(True)
	plt.xlabel('x1', fontsize = 15)
	plt.ylabel('y1', fontsize = 20)
	plt.title('Plot of X Points', fontsize = 25)

	#中心移动轨迹
	xx, yy =[],[]
	for centroid in centroids:
		xx.append(centroid[:,0])
		yy.append(centroid[:,1])

	plt.plot(xx, yy, 'rx--', markersize = 10)

#初始数据的可视化
#plotData(X,[init_centroids])
#plt.show()
#subX = [X]
#print(subX, len(subX), subX[0])

def runKmeans(X, centroids, max_iters):
	#K个 中心点
	K = len(centroids)
	centroids_all = []
	centroids_all.append(centroids)
	centroid_i = centroids
	for i in range(max_iters):
		idx = findClosestCentroids(X, centroid_i)
		centroid_i = computeCentroids(X, idx)
		centroids_all.append(centroid_i)
	return idx, centroids_all

idx, centroids_all = runKmeans(X, init_centroids, 20)
#plotData(X, centroids_all, idx)
#plt.show()

#===============================1.3 Random initialization

def initcentroids(X, K):
	m, n = X.shape
	idx = np.random.choice(m, K)
	centroids = X[idx]
	return centroids

for i in range(3):
	centroids = initcentroids(X, 3)
	idx, centroids_all = runKmeans(X, centroids, 10)
	#plotData(X, centroids_all, idx)
	#plt.show()


#===========================1.4 Image compresion with K-means

from skimage import io

A = io.imread('E:\python space\\2.jpg')
#print(A.shape)   (128,128,3)
#io.imshow(A)
#plt.show()
A = A/255

#find 16 clusters
X_picture = A.reshape(-1, 3)
#print(X_picture.shape)    #(16384, 3)
K = 16
centroids2 = initcentroids(X_picture, K)
idx, centroids2_all = runKmeans(X_picture, centroids2, 10)
#print(idx, centroids2_all)
img = np.zeros(X_picture.shape)
#最新一组中心点， 赋值给centroids
centroids2 = centroids2_all[-1]
#len(centroids2)  = 16

#给img每个点变为16个中心点之一
for i in range(len(centroids2)):
    img[idx == i] = centroids2[i]
print(img)

img = img.reshape((128, 128, 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(A)
axes[1].imshow(img)
plt.show()

