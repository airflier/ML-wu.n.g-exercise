import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#Scipy库中的optimiz可以实现matlab中fminunc的功能，来优化函数计算成本和梯度
import scipy.optimize as opt
#高级处理分类的一个信息report
from sklearn.metrics import classification_report as cr

#=================visualizing data
path = 'E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex2\ex2\ex2data2.txt'
data2 = pd.read_csv(path, names = ['Test1', 'Test2', 'Accepter'])
#print(data2.head())
def plot_data():
    positive = data2[data2.Accepter.isin([1])]
    negative = data2[data2.Accepter.isin([0])]
    fig ,ax = plt.subplots(figsize = (8,8))
    ax.scatter(positive['Test1'], positive['Test2'], s=50, c = 'b', marker = 'o', label = 'good')
    ax.scatter(negative['Test1'], negative['Test2'], s=50, c = 'r', marker = 'x', label = 'bad')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    ax.set_title('Data2')

plot_data()
#plt.show()

#=============feature mapping

#经过映射后，将只有两个特征的向量转换成为有28个特征的特征向量，故可以产生一个更复杂的边界
def feature_mapping(x1, x2, power):
    data={}
    #range()函数也不包括最后一个数
    for i in range(power+1):  
        for p in range(i+1):
            # format前面是'.' 不是','!!!!!!!!!!!!!!
           data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)
x1 = data2['Test1'].as_matrix()
x2 = data2['Test2'].as_matrix()
ndata2 = feature_mapping(x1, x2, power = 6)

#==============Regularized CostFunction

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta , X, Y):
    first = (-Y) * np.log(sigmoid(X @ theta))
    second = (1 - Y) * np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)
#X中已经包含1向量，故无需添加
X = ndata2.as_matrix()
Y = data2['Accepter'].as_matrix()
theta = np.zeros(X.shape[1]).reshape(-1)
#print(X.shape,Y.shape,theta.shape)
def costReg(theta, X, Y, l=1):
    ntheta = theta[1:, ]
    #补偿项
    reg = (l / (2*len(X))) * (ntheta @  ntheta.T)
    return cost(theta, X, Y) + reg
print(costReg(theta, X, Y, l=1))

#==============Regularized Gradient

def gradient(theta, x, y):
     return (x.T @ (sigmoid(x @ theta) -y)) / len(y)
def gradientReg(theta, x, y, l=1):
    reg = (1 / len(y)) * theta 
    reg[0] = 0 
    return gradient(theta, x, y) + reg 
#gradientReg(theta, X, Y, l=1)
#print(gradientReg(theta, X, Y))

#============Learning Parameters(theta)

'''
result2 = opt.fmin_tnc(func = costReg, x0 = theta, fprime = gradientReg, args=(X, Y, 2))
print(result2)
'''
#print(theta, theta.shape)
#print(X.shape, X, Y.shape, Y)
#逆矩阵操作
B = np.linalg.inv(X.T @ X)
final_theta = B @ X.T @ Y
print(final_theta, final_theta.shape)

'''高级库使用发放
from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.score(X, y)  # 0.8305084745762712
'''
#============Evaluating

def predict(theta, X):
     probability = sigmoid(X @ theta)
     return [1 if x >= 0.5 else 0 for x in probability]
'''
final_theta = result2[0]
'''
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a,b) in zip(predictions, Y)]
#准确度
acc = sum(correct) / len(correct)
print(acc)

#===========Decision Boundary
x = np.linspace(-1,1.5,250)
#生成xx yy坐标 利用meshgrid生成二维数组，在用ravel()变成一维数组
xx, yy = np.meshgrid(x, x)
z = feature_mapping(xx.ravel(), yy.ravel(), 6).as_matrix()
print(z.shape)
z = z @ final_theta
#z一定要和xx，yy保持一样的格式
print(z.shape,final_theta.shape,xx.shape)
z= z.reshape(xx.shape)
plot_data()
# 0 表示只画出一条线 默认画出z=0？？？
contour = plt.contour(xx, yy, z, 0 )
plt.clabel(contour,fontsize=10,colors=('k'))
plt.ylim(-0.8, 1.2)
plt.show()

