
#============Visulizing the dataset

#这一次将使用三部分的数据集training、validation、test
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#load data
path = 'E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex5\ex5\ex5data1.mat'
data = loadmat(path)
#Training set
X, y = data['X'], data['y']
#validation set 
Xval, yval = data['Xval'], data['yval']
#Test set
Xtest, ytest = data['Xtest'], data['ytest']
#Insert one column
X = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)
'''
print('X={},y={}'.format(X.shape, y.shape))
print('Xval={},yval={}'.format(Xval.shape, yval.shape))
print('Xtest={},ytest={}'.format(Xtest.shape, ytest.shape))

X=(12, 2),y=(12, 1)
Xval=(21, 2),yval=(21, 1)
Xtest=(21, 2),ytest=(21, 1)
'''
def plotData():
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:,1:], y, c='r', marker='x')
    plt.xlabel('Change in water level(X)')
    plt.ylabel('water flowing out of the dam(y)')
    plt.grid(True)
 
#plotData()


#=====================Regularized linear regression cost function

def costReg(theta, X, y, l):
    '''
    X (12 ,2) theta(2,) y(12, 1)
    '''
    cost = ((X @ theta - y.flatten()) ** 2).sum()
    regterm = l * (theta[1:] @ theta[1:])
    return (cost  + regterm )/ (2 * len(X))

theta = np.ones(X.shape[1])
#print(costReg(theta, X, y, 1))

#====================Regularized linear regression gradient

def gradientReg(theta, X, y, l):
    #grad has same shape with theta (2,)
    grad = X.T @ (X @ theta - y.flatten()) 
    regterm = l * theta
    regterm[0] = 0
    return (grad + regterm) / len(X)
print(gradientReg(theta, X, y, 1))
#[-15.30301567 598.25074417]

#===================fitting

def train(X, y, l):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun = costReg, x0 = theta, args=(X, y, l), method = 'TNC', jac = gradientReg)
    return res.x
fit_theta = train(X, y, 0)
plotData()
#plt.plot(X[:,1], X @ fit_theta)
#plt.show()

#===================Learning curves

def plot_learning_curve(X, y, Xval, yval, l):
#学习曲线即交叉验证误差和训练误差
    xx = range(1, len(X)+1)
    training_cost, cv_cost = [],[]
    for i in xx:
        res = train(X[: i], y[:i], l)
        training_cost_i = costReg(res, X[:i], y[:i], 0)
        cv_cost_i = costReg(res, Xval, yval, 0)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)

    plt.figure(figsize=(8, 5))
    plt.plot(xx, training_cost, label = 'training cost')
    plt.plot(xx, cv_cost, label = 'cv cost')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning curve for linear regression')
    plt.grid(True)
#plot_learning_curve(X, y, Xval, yval, 0)
#plt.show()

#====================Learnign polynomial Regression

def genPolyFeatures(X, power):
    #增加多项式特征，每次在array的最后一列插入第二列的i+2次方，2次方开始
    Xpoly = X.copy()
    for i in range(2, power + 1):
        Xpoly = np.insert(Xpoly, Xpoly.shape[1], np.power(Xpoly[:,1],i), axis = 1)
    return Xpoly

def get_means_std(X):
    #获取训练集的均值和误差，来标准化数据
    means = np.mean(X, axis=0)
    #ddof=1 表示样本标准差使用np.std()时，将ddof=1则是样本标准差，默认=0是总体标准差。而pandas默认计算样本标准差。
    stds= np.std(X, axis=0, ddof=1)
    return means, stds

def featureNormalize(myX, means, stds):
    #standarlize
    X_norm = myX.copy()
    X_norm[:,1:] = X_norm[:, 1:] - means[1:]
    X_norm[:,1:] = X_norm[:, 1:] / stds[1:]
    return X_norm

#print(genPolyFeatures(X, 6).shape, genPolyFeatures(X, 6))　（12, 7)
train_means, train_stds = get_means_std(genPolyFeatures(X, 6))
#print(train_means.shape, train_stds.shape) (7, ) (7, )
X_norm = featureNormalize(genPolyFeatures(X, 6), train_means, train_stds)
Xval_norm = featureNormalize(genPolyFeatures(Xval, 6), train_means, train_stds)
Xtest_norm = featureNormalize(genPolyFeatures(Xtest, 6), train_means, train_stds)

def plot_polofit(means, stds, l):
    theta = train(X_norm, y, l) #（7, ）
    x = np.linspace(-75,55,50) #50个元素
    xmat = x.reshape(-1,1)
    xmat = np.insert(xmat, 0 ,1, axis=1)
    Xmat = genPolyFeatures(xmat, 6)
    Xmat_norm = featureNormalize(Xmat, means, stds)
    print(Xmat.shape, Xmat_norm.shape)
    yy = Xmat_norm @ theta
   # print(yy.shape)
    plotData()
    plt.plot(x, yy)
#plot_polofit(train_means, train_stds, 0)
#plot_learning_curve(X_norm, y, Xval_norm, yval, 0)
#plt.show()

#===================adjusting lambda(the regularization parameter)

#lambada = 0 overfitting 
#try : lambda = 1
'''
plot_polofit(train_means, train_stds, 1)
plot_learning_curve(X_norm, y, Xval_norm, yval, 1)
plt.show()
'''

#==============selecting lambda using a cv set
lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
errors_train, errors_val = [], []
for l in lambdas:
    theta = train(X_norm, y, l)
    errors_train.append(costReg(theta, X_norm, y, 0))
    errors_val.append(costReg(theta, Xval_norm, yval, 0))

plt.figure(figsize=(8,5))
plt.plot(lambdas, errors_train,label = 'Train')
plt.plot(lambdas, errors_val, label = 'Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.grid(True)
#plt.show()
print(lambdas[np.argmin(errors_val)])


plot_polofit(train_means, train_stds, 3)
plot_learning_curve(X_norm, y, Xval_norm, yval, 3)
plt.show()

theta = train(X_norm, y, 3)
print('test cost(lambda = {}) = {}'.format(3, costReg(theta, Xtest_norm, ytest, 0)))






