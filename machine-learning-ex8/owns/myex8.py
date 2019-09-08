import pandas as pd 
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\ex8data1.mat')
X = mat['X']
Xval, yval = mat['Xval'], mat['yval']
print(X.shape, Xval.shape, yval.shape)
#(307, 2) (307, 2) (307, 1)

def plot_data():
    plt.figure(figsize = (8, 5))
    plt.plot(X[:,0],X[:,1], 'x')

plot_data()
#plt.show()

#==============1.1 Gaussian distribution

def gaussian(X, mu, sigma2):
    #output a m-dimension vector,including the samples' probability
    '''
    #利用矩阵解法
    n = len(mu)
    #若sigma2 是向量，则将其转换为协方差矩阵
    
    if sigma2.ndim == 1 or (sigma2.ndim == 2 and (sigma2.shape[1] == 1 or sigma2.shape[0] == 1)):
        sigma2 = np.diag(sigma2)
    
    X = X - mu
    p1 = np.power(2 * np.pi, -n/2)*np.sqrt(np.linalg.det(sigma2))
    e = np.diag(X @ np.linalg.inv(sigma2) @ X.T)   #取对角线？、
    p2 = np.exp(-0.5 * e)

    return p1 * p2
'''
    #'for' solution
    m, n = X.shape
    if sigma2.ndim == 1 or (sigma2.ndim == 2 and (sigma2.shape[1] == 1 or sigma2.shape[0] == 1)):
        sigma2 = np.diag(sigma2)
    norm = 1./(np.power((2*np.pi), n/2) * np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm * exp

#==================Estimating parameters for Gaussian

def gaussianParameters(X, useMultivariate):
    mu = X.mean(axis = 0)
    if useMultivariate:
        sigma2 = ((X-mu).T @ (X - mu)) / len(X)
    else:
        sigma2 = X.var(axis = 0, ddof = 0) #/m 而不是m-1
    return mu,sigma2

def plotContours(mu, sigma2):
    delta = .3
    x = np.arange(0,30,delta)
    y = np.arange(0, 30, delta) 

    xx, yy = np.meshgrid(x,y)
    points = np.c_[xx.ravel(), yy.ravel()]
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)

    cont_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx, yy, z, cont_levels)  # 这个levels是作业里面给的参考,或者通过求解的概率推出来。

    plt.title('Gaussian Contours',fontsize=16)

# First contours without using multivariate gaussian:
plot_data()
useMV = False
plotContours(*gaussianParameters(X, useMV))

# Then contours with multivariate gaussian:
plot_data()
useMV = True 
# *表示解元组
plotContours(*gaussianParameters(X, useMV))
'''
plt.show()
'''
#a = gaussian(X, *gaussianParameters(X, useMV))
#print(a.max(), a.min(), a.mean())
#===================1.3selecting the threshold

def selectThreshold(yval, pval):
    def computeF1(yval, pval):
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0 
        F1 = 2 * prec * rec/(prec + rec) if (prec + rec) else 0
        return F1
    
    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestepsilon = 0, 0
    for e in epsilons:
        pval_ = pval < e #不满足返回false一样占据该list的位置
        nowF1 = computeF1(yval, pval_)
        if nowF1 > bestF1:
            bestF1 = nowF1
            bestepsilon = e

    return bestF1, bestepsilon

mu, sigma2 = gaussianParameters(X, useMultivariate=False)
pval = gaussian(Xval, mu, sigma2)
bestF1, bestEpsilon = selectThreshold(yval, pval) 
#print(bestF1, bestEpsilon) #0.8750000000000001 [8.99985263e-05]

#========捕捉异常点

y = gaussian(X, mu, sigma2)
xx = np.array([X[i] for i in range(len(y)) if y[i] < bestEpsilon])
#实际画出该点即是一种标记方式
plt.scatter(xx[:,0], xx[:,1], s= 80, facecolors = 'none', edgecolors='r')
#plt.show()

#============High dimensional dataset

mat2 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\ex8data2.mat')
X2, Xval2, yval2 = mat2['X'], mat2['Xval'], mat2['yval']
#print(X2.shape, Xval2.shape, yval2.shape)
#(1000, 11) (100, 11) (100, 1)

mu2,sigma22 = gaussianParameters(X2, useMultivariate=False)
ypred2 = gaussian(X2, mu2, sigma22)
yval2pred = gaussian(Xval2, mu2, sigma22)

bestF1_2, bestEpsilon_2 = selectThreshold(yval2, yval2pred)
anoms = [X2[i] for i in range(len(X2)) if ypred2[i] < bestEpsilon_2]
#print(bestEpsilon_2, len(anoms))
#[1.3786075e-18] 117


