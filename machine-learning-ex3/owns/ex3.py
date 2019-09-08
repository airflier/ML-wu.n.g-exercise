import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#=====================Dataset

#数据为mat格式 采用Scipy.io的loadmat函数
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

X,y = load_data('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex3\ex3\ex3data1.mat')
print(y, X.shape, y.shape, np.unique(y))

#====================visualizing the data

def plot_an_image(X):
    pick_one = np.random.randint(0,5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize = (1, 1))
    #绘制矩阵的函数，参数 矩阵 ；cmap颜色映射方式
    ax.matshow(image.reshape(20, 20), cmap = 'gray_r')
    #这种空写法，将不显示数轴刻度
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('this number shoule be {}'.format(y[pick_one]))
#plot_an_image(X)

#随机画100个数

def plot_100_image(X):
    #从[0,X.shape[0]])中，随机选100个样本
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]   #(100,400)
    #注意subplots和subplot的区别
    fig, ax_array = plt.subplots(nrows = 10, ncols =10, figsize=(10, 10), sharex=True, sharey = True)
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row +column].reshape(20,20), cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

#plot_100_image(X)

#=====================Vectorizing the cost function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def reg_cost(theta, X, y, l):  #y(5000, 1) X(5000, 400)
    #l 为 lambda 惩罚项系数
    #对第theta0不惩罚
    thetaReg = theta[1:]
    first = (-y * np.log(sigmoid(X @ theta))) + (y-1) * np.log(1 - sigmoid( X @ theta))
    reg = (l/(2 * len(X))) * (thetaReg @ thetaReg.T)
    return np.mean(first) +reg
'''
theta = np.zeros(400)
thetaReg = theta[1:]
a = thetaReg @ thetaReg.T
b = thetaReg * thetaReg
print("a=",a,'a.shape=',a.shape, b, b.shape)
'''

#=================Vectorizing the gradient


def reg_gradient(theta, X, y, l):
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    #数组拼接 加入一个0
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    return first + reg

#==================One vs all classification

from scipy.optimize import minimize
#插入一列，矩阵；插入行列；插入值；横向/纵向    
X = np.insert(X, 0, 1, axis = 1)
def one_vs_all(X, y, l, K):
    '''
    X: (m, n+1) with x0 =1
    y; (m,)
    l: constant for regularization
    K: numbel of labels
    return:　 trained parameters
    '''
    all_theta = np.zeros((K, X.shape[1])) #(10, 401)
    for i in range(1, K+1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])
        ret = minimize(fun = reg_cost, x0 = theta, args=(X, y_i, l),method='TNC',
                      jac = reg_gradient, options={'disp': True})
        #ret.x x代表解决的矩阵 ，success bool量代表是否成功，message描述终结信息
        all_theta[i-1, :] =ret.x
    return all_theta   #(10,401)
#X (5000,401)


def predict_all(X, all_theta):
    #计算出每类的可能性( 5000,10)
    h = sigmoid(X @ all_theta.T) 
    #找出每行最大值的下标
    h_argmax = np.argmax(h, axis = 1)
    #返回5000个样本对应的预测值 +1是因为数组从0开始
    return h_argmax + 1
'''
    y = y.flatten()
final_theta = one_vs_all(X, y, 1, 10)
y_pred = predict_all(X, final_theta)

accuracy = np.mean(y_pred == y)
print('accuracy = {0}%'.format(accuracy * 100))
'''
#===========Neural Networks

#导入已经训练好的权重
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']
theta1, theta2 = load_weight('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex3\ex3\ex3weights.mat')
#theta1(25,401) theta2(10,46)
#X(5000,401)
z2 = X @ theta1.T
#z2 (5000,25)
a2 = sigmoid(z2)
#a2 (5000,26)
a2 = np.insert(a2, 0, 1, axis = 1)
#z3 (5000,10)
z3 = a2 @ theta2.T 
a3 = sigmoid(z3)
#argmax 返回的坐标
ny_pred = np.argmax(a3, axis = 1) + 1
#一定要保证y_pred的格式与y的格式一致
ny_pred = ny_pred.reshape(5000,1)
naccuracy = np.mean(ny_pred == y)
print("naccuracy = {0}%".format(naccuracy * 100))



