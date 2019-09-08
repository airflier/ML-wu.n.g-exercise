
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report

#=======================visualizing the data===

def load_mat(path):
    #读取数据
    data = loadmat(path)
    X = data['X']
    y = data['y'].flatten()
    return X, y


def plot_100_images(X):
    #随机画100个数字
    index = np.random.choice(range(X.shape[0]), 100) #shape[0] = 5000
    images = X[index, :] #image.shape = (100, 400)
    print("images.type", images.shape)
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), sharex = True, sharey = True)
    for r in range(10):
        for c in range(10):
            #窗口显示矩阵
            ax[r,c].matshow((images[r*10 + c]).reshape(20, 20), cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

X, y = load_mat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex4\ex4\ex4data1.mat')
#print("X.shape", X.shape[0])
#plot_100_images(X)

#======================load train data set and weight

#将相应的标签转换成为对应的向量
'''
使用sklearn函数
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse  = False ) #return a array
y_onehot  = encoder.fit_transform(y.reshape(-1,1))
return y_onehot 
'''

def expand_y(y):
    result = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i-1] = 1
        result.append(y_array)
    #将list转换成为array    
    return np.array(result)
'''
    X = np.insert(X, 0, 1, axis = 1)
y = expand_y(y)
#print(X.shape, y.shape)

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex4\ex4\ex4weights.mat')
#print(theta1.shape, theta2.shape) (25, 401)(10, 26)

#使用 高级的优化神经网络时，需要将多个参数矩阵展开
def serialize(a, b):
    #np.r_ 为array添加行
    return np.r_[a.flatten(), b.flatten()]
theta_expand = serialize(theta1, theta2)
print(theta_expand.shape)

#恢复形状
def deserialize(seq):
    return seq[0 : 25*401].reshape(25, 401), seq[25*401:].reshape(10,26)

#=====================Feedforward and cost function

def sigmoid(z):
    return 1 /  (1 + np.exp(-z))
def feed_forward(theta1, theta2, X):
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis = 1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3
a1, z2, a2, z3, h = feed_forward(theta1, theta2, X)

#print(y.shape, y[1], y[1].shape, h[1].shape, y[1] @ np.log(h[1].T)) # (5000,10) (10,) (10,)
def cost(theta1,theta2, X, y):
    a1, z2, a2, z3, h = feed_forward(theta1, theta2, X)
    J = 0
    for i in range(len(X)):
        first = -y[i] @ np.log(h[i]).T
        second = (1 - y[i]) @ np.log(1 - h[i])
        J = (J + first - second) 
    J = J / len(X)

    使用矢量化方法不用循环
    J = -y * np.log(h) - (1-y) * np.log(1 - h)
    return J.sum() / len(X)

    return J

cost(theta1, theta2, X, y)

#==============Regularize cost function

def regularized_cost(theta, X, y, lamda=1):
    t1, t2 = deserialize(theta_expand)
    reg  = np.sum(t1[: , 1:] ** 2) + np.sum(t2[: , 1:] ** 2) 
    return lamda / (2 * len(X)) * reg + cost(theta1, theta2, X, y)

regcost = regularized_cost(theta_expand, X, y, 1)
#print(regcost) 0.3837698590909233

#================Backpropagation(反向传播)

#sigmoid函数的求导
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

#随机初始化参数
def random_init(size):
    #从服从的均匀分布的范围中随机返回size大小的值
    return np.random.uniform(-0.12, 0.12, size)

#反向传播，来获得整个网络代价函数的梯度，以便于在优化算法求解
#注意理解各个参数的维度
#print("a1", a1.shape, "z2", z2.shape, "a2", a2.shape, "z3", z3.shape,"h", h.shape)
#a1 (5000, 401) z2 (5000, 25) a2 (5000, 26) z3 (5000, 10) h (5000, 10)
#theta1 (25, 401) theta2(10, 26)
def gradient(theta1, theta2, X, y):
    #return 所有theta的梯度，故D（i）的维度应该与theta一致
    d3 = h - y #(5000, 10)
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2) #(5000,25)
    D2 = d3.T @ a2  # =D(theta2)=(10,26)
    D1 = d2.T @ a1  #=(25, 401)
    D = (1 / len(X)) * serialize(D1, D2) #(10285, )
    return D, D1, D2
D, D1, D2 = gradient(theta1, theta2, X, y)


#================Regularized Neural NetWorks

def regularized_gradient(theta1,theta2, D1, D2, X, y, lamda=1):
    theta1[:, 0] = 0
    theta2[:, 0] = 0        
    reg_D1 = D1 + (lamda / len(X)) * theta1
    reg_D2 = D2 + (lamda / len(X)) * theta2
    return serialize(reg_D1, reg_D2)



#===================Gradient Checking
def gradient_checking(theta, theta1, theta2, D1, D2, X, y, e):
    def a_numeric_grad(plus, minus):
        #对每个参数theta_i计算理论上的梯度
        return (regularized_cost(plus, X, y) - regularized_cost(minus,X, y)) / (2 * e)
    numeric_grad = []
    for i in range(len(theta)): #（10285，）
        plus = theta.copy()
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)
    print(len(numeric_grad))

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(theta1,theta2, D1, D2, X, y, lamda=1)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)
    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

gradient_checking(theta_expand, theta1, theta2, D1, D2, X, y, e=0.0001)


#==================learning parameters using fmincg

def nn_traning(X,y):
    #利用之前函数随机初始化参数
    init_theta = random_init(10285)

    res = opt.minimize(fun = regularized_cost, x0 = init_theta,
                        args = (X,y,1), method = 'TNc',
                        jac = regularized_gradient, options = {'maxiter':400})
    return res
res = nn_traning(X, y)
print(res)

def accuracy(theta, X, y):
    t1, t2 = deserialize(res.x)
    a, b, c, d, h = feed_forward(t1, t2, X)
    y_pred = np.argmax(h, axis = 1) + 1
    print(classification_report(y, y_pred))
accuracy(res.x, X, y)

#================可视化隐藏层
def plot_hidden(theta):
    t1, t2 = deserialize(theta)
    t1 = t1[:,1:]
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8,8))
    for r in range(5):
        for c in range(5):
            ax[r,c].matshow(t1[r*5 + c].reshape(20, 20), cmap = 'gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()
plot_hidden(res.x)
'''

raw_X, raw_y = load_mat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex4\ex4\ex4data1.mat')
X = np.insert(raw_X, 0, 1, axis=1)
y = expand_y(raw_y)
X.shape, y.shape
'''
((5000, 401), (5000, 10))
'''
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2'] 
t1, t2 = load_weight('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex4\ex4\ex4weights.mat')
t1.shape, t2.shape
# ((25, 401), (10, 26))
def serialize(a, b):
    '''展开参数'''
    return np.r_[a.flatten(),b.flatten()]
theta = serialize(t1, t2)  # 扁平化参数，25*401+10*26=10285
theta.shape  # (10285,)
def deserialize(seq):
    '''提取参数'''
    return seq[:25*401].reshape(25, 401), seq[25*401:].reshape(10, 26)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def feed_forward(theta, X,):
    '''得到每层的输入和输出'''
    t1, t2 = deserialize(theta)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = X
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)
    
    return a1, z2, a2, z3, a3 
a1, z2, a2, z3, h = feed_forward(theta, X)
def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    return J
'''
     # or just use verctorization
     J = - y * np.log(h) - (1 - y) * np.log(1 - h)
     return J.sum() / len(X)
'''
cost(theta, X, y)  # 0.2876291651613189
def regularized_cost(theta, X, y, l=1):
    '''正则化时忽略每层的偏置项，也就是参数矩阵的第一列'''
    t1, t2 = deserialize(theta)
    reg = np.sum(t1[:,1:] ** 2) + np.sum(t2[:,1:] ** 2)  # or use np.power(a, 2)
    return l / (2 * len(X)) * reg + cost(theta, X, y)
regularized_cost(theta, X, y, 1)  # 0.38376985909092354
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
def random_init(size):
    '''从服从的均匀分布的范围中随机返回size大小的值'''
    return np.random.uniform(-0.12, 0.12, size)
print('a1', a1.shape,'t1', t1.shape)
print('z2', z2.shape)
print('a2', a2.shape, 't2', t2.shape)
print('z3', z3.shape)
print('a3', h.shape)
'''
a1 (5000, 401) t1 (25, 401)
z2 (5000, 25)
a2 (5000, 26) t2 (10, 26)
z3 (5000, 10)
a3 (5000, 10)
'''
def gradient(theta, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error 
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y # (5000, 10)
    d2 = d3 @ t2[:,1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = d3.T @ a2  # (10, 26)
    D1 = d2.T @ a1 # (25, 401)
    D = (1 / len(X)) * serialize(D1, D2)  # (10285,)
    
    return D
def regularized_gradient(theta, X, y, l=1):
    """不惩罚偏置单元的参数"""
    t1,t2=deserialize(theta.copy())
    a1, z2, a2, z3, h = feed_forward(theta, X)
    D1, D2 = deserialize(gradient(theta, X, y))
    t1[:,0] = 0
    t2[:,0] = 0
    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2
    
    return serialize(reg_D1, reg_D2)

def gradient_checking(theta, X, y, e):
    def a_numeric_grad(plus, minus):
        """
        对每个参数theta_i计算数值梯度，即理论梯度。
        """
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (e * 2)
   
    numeric_grad = [] 
    for i in range(len(theta)):
        plus = theta.copy()  # deep copy otherwise you will change the raw theta
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)
    
    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))
#gradient_checking(theta, X, y, e= 0.0001)#这个运行很慢，谨慎运行

def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 500})
    return res

res = nn_training(X, y)#慢
print(res)
'''
     fun: 0.5156784004838036
     jac: array([-2.51032294e-04, -2.11248326e-12,  4.38829369e-13, ...,
        9.88299811e-05, -2.59923586e-03, -8.52351187e-04])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 271
     nit: 17
  status: 1
 success: True
       x: array([ 0.58440213, -0.02013683,  0.1118854 , ..., -2.8959637 ,
        1.85893941, -2.78756836])
'''
def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))

#accuracy(res.x, X, raw_y)
def plot_hidden(theta):
    t1, _ = deserialize(theta)
    t1 = t1[:, 1:]
    fig,ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6,6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()
plot_hidden(res.x)