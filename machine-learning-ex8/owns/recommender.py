import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\ex8_movies.mat')
#print(mat.keys())
Y, R = mat['Y'], mat['R']
#print(Y.shape, R.shape)
#(1682, 943) (1682, 943)
#print(Y[:15], R[:15])
nm, nu = Y.shape
nf = 100
#第一个电影的评分
# print(Y[0].sum() / R[0].sum())

#visualize the ratings matrix
fig = plt.figure(figsize = (8, 8*(1682./943.)))
plt.imshow(Y, cmap = 'rainbow')
plt.colorbar()
plt.ylabel('movies(%d)'%nm, fontsize = 20)
plt.xlabel('users(%d)'%nu, fontsize = 20)
#plt.show()

#==========collaborative filtering learning algorithm

mat2 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\ex8_movieParams.mat')
print(mat2.keys())
X = mat2['X']
Theta = mat2['Theta']
numusers = int(mat2['num_users'])
nummovies = int(mat2['num_movies'])
numfeatures = int(mat2['num_features'])
#print(numusers, nummovies, numfeatures)
#       943          1682       10
#reduce the data
numusers = 4; nummovies = 5; numfeatures = 3
X = X[:nummovies, :numfeatures] #(5, 3)
Theta = Theta[:numusers, :numfeatures] #(4, 3)
Y = Y[:nummovies, :numusers]
R = R[:nummovies, :numusers]
#print(X, Theta)
#========collaborative cost function

#展开参数
def serialize(X, Theta):
    return np.r_[X.flatten(), Theta.flatten()]
#print(serialize(X, Theta)) (27, )
#提取参数
def deserialize(seq, nm, nu, nf):
    return seq[:nm*nf].reshape(nm, nf), seq[nm*nf:].reshape(nu, nf)

def collCostfunction(params, Y, R, nm, nu, nf, l=0):
    """
    params : 拉成一维之后的参数向量(X, Theta)
    Y : 评分矩阵 (nm, nu)
    R ：0-1矩阵，表示用户对某一电影有无评分
    nu : 用户数量
    nm : 电影数量
    nf : 自定义的特征的维度
    l : lambda for regularization
    """
    X, Theta = deserialize(params, nm, nu, nf)
    
    # (X@Theta)*R含义如下： 因为X@Theta是我们用自定义参数算的评分，但是有些电影本来是没有人
    # 评分的，存储在R中，0-1表示。将这两个相乘，得到的值就是我们要的已经被评分过的电影的预测分数。
    error = 0.5*np.square((X@Theta.T - Y)*R).sum()
    reg1 = .5*l*np.square(Theta).sum()
    reg2 = .5*l*np.square(X).sum()
    
    return error + reg1 +reg2

'''
def collCostfunction(X, Theta, Y, R, nm, nu, nf, l = 0):
    #parameters : 一维(X, Theta)
    error = 0.5 * np.square((X @ Theta.T - Y) * R).sum()
    reg1 = 0.5 * l * np.square(Theta).sum()
    reg2 = 0.5 * l * np.square(X).sum()
    return error + reg1 + reg2
'''

#print(collCostfunction(X, Theta, Y, R, nummovies, numusers, numfeatures, l = 0))
#print(collCostfunction(X, Theta, Y, R, nummovies, numusers, numfeatures, l = 1.5))
'''
22.224603725685675
31.34405624427422
'''

#================collaborative filtering gradient

def collGradient(params, Y, R, nm, nu, nf, l=0):
    """
    计算X和Theta的梯度，并序列化输出。
    """
    X, Theta = deserialize(params, nm, nu, nf)
    
    X_grad = ((X@Theta.T-Y)*R)@Theta + l*X
    Theta_grad = ((X@Theta.T-Y)*R).T@X + l*Theta
    
    return serialize(X_grad, Theta_grad)

'''
def collGradient(X, theta, Y, R, nm, nu, nf, l = 0):
    X_grad = ((X @ theta.T - Y) * R) @ theta +l * X
    theta_grad = ((X @ theta.T - Y) * R).T @ X + l * theta
    return serialize(X_grad, theta_grad)
'''
def checkGradient(params, Y, myR, nm, nu, nf, l = 0.):
    """
    Let's check my gradient computation real quick
    """
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    
    # 分析出来的梯度
    grad = cofiGradient(params,Y,myR,nm,nu,nf,l)
    
    # 用 微小的e 来计算数值梯度。
    e = 0.0001
    nparams = len(params)
    e_vec = np.zeros(nparams)

    # Choose 10 random elements of param vector and compute the numerical gradient
    # 每次只能改变e_vec中的一个值，并在计算完数值梯度后要还原。
    for i in range(10):
        idx = np.random.randint(0,nparams)
        e_vec[idx] = e
        loss1 = cofiCostFunc(params-e_vec,Y,myR,nm,nu,nf,l)
        loss2 = cofiCostFunc(params+e_vec,Y,myR,nm,nu,nf,l)
        numgrad = (loss2 - loss1) / (2*e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(numgrad, grad[idx], diff))

'''
def checkGradient(X, theta, Y, myR, nm, nu, nf, l = 0):
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    
    #计算得来的梯度
    grad = collGradient(X, theta, Y, myR, nm, nu, nf, l)

    # small e compute
    e = 0.0001
    parameters = serialize(X, theta)
    nparams = len(parameters)
    e_vec = np.zeros(nparams)

    #select 10 random elements
    for i in range(10):
        idx = np.random.randint(0, nparams)
        e_vec[idx] = e
        Xn, thetan = deserialize(parameters - e_vec, nm, nu, nf)
        Xn1, thetan1 = deserialize(parameters + e_vec, nm, nu, nf)
        loss1 = collCostfunction(Xn, thetan, Y, myR, nm, nu, nf, l)
        loss2 = collCostfunction(Xn1, thetan1, Y, myR, nm, nu, nf, l)
        numgrad = (loss2 - loss1) / (2 * e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(numgrad, grad[idx], diff))
'''
#print("check with lambda = 0...")
#print(checkGradient(X, Theta, Y, R, nummovies, numusers, numfeatures))
#print("check with lambda = 1.5...")
#print(checkGradient(X, Theta, Y, R, nummovies, numusers, numfeatures, l = 1.5))

#=================p========Learning movie rem=commmendations

movies = [] 
#注意编码 改成'utf-8'
with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\movie_ids.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        #就是把txt去掉前面的电影序号，变成list
        movies.append(' '.join(line.strip().split(' ')[1:]))

my_ratings = np.zeros((1682, 1))

my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

'''
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(my_ratings[i], movies[i])
'''


#import Y and R
mat3 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\ex8_movies.mat')
#print(mat3.keys())
Y2, R2 = mat3["Y"], mat3['R']
#print(Y2.shape, R2.shape)
#(1682, 943) (1682, 943)
Y2 = np.c_[Y2, my_ratings]
R2 = np.c_[R2, my_ratings]-
nm, nu = Y2.shape
nf = 10

#=======================normalize

def nomarlize(Y, R):
    Ymean = (Y.sum(axis = 1) / R.sum(axis = 1)).reshape(-1,1)
    #use R to delete the unrated movies' scores
    Ynorm = (Y - Ymean) * R
    return Ynorm, Ymean

Y2norm, Y2mean = nomarlize(Y2, R2)
#print(Ynorm.shape, Ymean.shape)
#(1682, 944) (1682, 1)

X2 = np.random.random((nm, nf))
Theta2 = np.random.random((nu, nf))
params2 = serialize(X2, Theta2)
l = 10

import scipy.optimize as opt
res = opt.minimize(fun = collCostfunction,
                    x0 = params2,
                    args = (Y2norm, R2, nm, nu, nf, l),

                    method = 'TNC',
                    jac  = collGradient,
                    options = {'maxiter': 100})

res = res.x #(26260, )
fit_X, fit_Theta = deserialize(res, nm, nu, nf)
#print(fit_Theta.shape,fit_X.shape) 
#(944, 10) (1682, 10)
#训练完毕！！

#================prediction

pred_mat = fit_X @ fit_Theta.T 
#print(pred_mat.shape)  （1682， 944）
pred = pred_mat[:, -1] + Y2mean.flatten()
#print(pred.shape)
#(1682, )
pred_sorted_idx = np.argsort(pred)[::-1]
print(pred_sorted_idx)

print("Top recommendations:")
for i in range(10):
    print("predicting rating %0.1f for movies %s."
    %(pred[pred_sorted_idx[i]], movies[pred_sorted_idx[i]]))

print("\nOrifinal ratings:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.'% (my_ratings[i],movies[i]))