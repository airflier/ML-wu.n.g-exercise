        #Support Vector Machines

#===================Example Dataset1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat 
from sklearn import svm
#大多数SVM库会自动添加额外特征 x0 theta0,
mat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\ex6data1.mat')
print(mat.keys())
#打印数据的key
#dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
X = mat['X'] #(51, 2)
y = mat['y'] #(51, 1)
#print(X.shape, y.shape)

yn = list(enumerate(y))
ylist1 = [i for i, x in yn if x == 1]  #21
ylist0 = [i for i, x in yn if x == 0]  #30
#print(len(ylist1), len(ylist0))

def plotData(X, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X[ylist1,0], X[ylist1,1], c=ylist1, marker = 'o', cmap = 'rainbow', label = 'Spam')
    plt.scatter(X[ylist0,0], X[ylist0,1], c=ylist0, marker = 'x', cmap = 'rainbow', label = 'NoSpam')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'best') #画图时要有label
    #plt.show( )
#plotData(X, y)

def plotBoundary(clf, X):
    x_min, x_max = X[:, 0].min() * 1.2, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 1.1, X[:, 1].max() * 1.1 
    #形成xx ,yy 为坐标的网格
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    #np.c_ 到底形成啥 ？？？
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)

models = [svm.SVC(C, kernel = 'linear') for C in [1, 100]]
#拟合模型
clfs = [model.fit(X, y.ravel()) for model in models]

title = ['SVM Decision Boundary with c = {} (Example Dataset 1)'.format(C) for C in [1,100] ]
'''
for model, title in zip(clfs, title):
    #plt.figure(figsize=(8, 5))
    plotData(X, y)
    plotBoundary(model, X)
    plt.title(title)
    #plt.show()
'''

#===================Gaussian Kernel

def gaussKernel(x1, x2, sigma):
        return np.exp(- ((x1 - x2) ** 2).sum() / (2 * sigma ** 2))

print(gaussKernel(np.array([1, 2, 1]), np.array([0, 4, -1]), 2.))

#=======================Example Dataset2

mat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\ex6data2.mat')
#print(mat.keys())
X2 = mat['X']
y2 = mat['y']
yn2 = list(enumerate(y2))
y2list1 = [i for i, x in yn2 if x == 1]  #480
y2list0 = [i for i, x in yn2 if x == 0]  #383
#print(len(y2list0), len(y2list1))
def plotData2(X, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X2[y2list1,0], X2[y2list1,1], c=y2list1, marker = 'o', cmap = 'rainbow', label = 'Spam')
    plt.scatter(X2[y2list0,0], X2[y2list0,1], c=y2list0, marker = 'x', cmap = 'rainbow', label = 'NoSpam')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'best')
#plotData2(X2,y2)
#plt.show()
sigma = 0.1
gamma = np.power(sigma, -2.)/2
clf = svm.SVC(C=100, kernel = 'rbf', gamma = gamma)
model = clf.fit(X2, y2.flatten())
#plotBoundary(model, X2)
#plt.show()

#====================Example Dataset 3

mat3 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\ex6data3.mat')
X3, y3 = mat3['X'], mat3['y']
Xval, yval = mat3['Xval'], mat3['yval']
yn3 = list(enumerate(y3))
y3list1 = [i for i, x in yn3 if x == 1]  #106
y3list0 = [i for i, x in yn3 if x == 0]  #105
print(len(y3list0), len(y3list1))
def plotData3(X, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X3[y3list1,0], X3[y3list1,1], c=y3list1, marker = 'o', cmap = 'rainbow', label = 'Spam')
    plt.scatter(X3[y3list0,0], X3[y3list0,1], c=y3list0, marker = 'x', cmap = 'rainbow', label = 'NoSpam')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'best')
    
#plotData3(X3, y3)
#plt.show()

Cvalues = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for C in Cvalues:
    for sigma in sigmavalues:
        model = svm.SVC(C = C, kernel = 'rbf', gamma = gamma)
        model.fit(X3, y3.flatten())
        # model.score()评价函数
        this_score = model.score(Xval, yval)
        if this_score > best_score:
            best_score = this_score
            best_pair = (C, sigma)
print('best_pair={}, best_score={}'.format(best_pair, best_score))
#BEST pair = （1.0, 0.01), best_score = 0.965
model  = svm.SVC(C = 1, kernel = 'rbf', gamma = np.power(0.1, -2.) / 2 )
model.fit(X3, y3.flatten())
plotData3(X3, y3)
plotBoundary(model, X3)
plt.show()


