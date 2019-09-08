#matplotlib inline 该命令在jupyter notebook使用，调用plt可以在控制台生成画布

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lm


#==============数据导入及数据可视化

path = 'E:\lessons\ML wu.n.g\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])
#DataFrame.insert(loc,column,value) loc为列数 column列名
data.insert(0,'Ones',1)
#DataFrame.head默认输出前5行数据，可以主动改数
print(data.head())
#DataFrame.describe 对数据的描述  （可选三个参数，1.百分比 2.include=['O']会统计离散型）
#std 标准差
print(data.describe())
#使用散点图描绘数据 figsize图片大小inches
data.plot(kind = 'scatter',x = 'Population',y = 'Profit', figsize=(8,5))
plt.show()


#==============cost function

def computeCost(X,y,theta):
        #np.power(x1,x2) x1,x2可以是数字或数组，只需保证x1和x2列数相同
        #.T 代表转置
        deviation = np.power(((X*theta) - y),2)
        return np.sum(deviation) / (2 * len(X))
#变量初始化 X(training data) y(target value)
# data.shap[0]表示行数
cols = data.shape[1]
#pandas中 data.iloc[x:x,y:y]根据行列号 数字 索引
#取前cols-1列
X = data.iloc[:,0:cols - 1] 
y = data.iloc[:,cols-1:]
#print(type(X))
#print(type(y))
#注意matrix 和 array 的区别;同时需要注意转化X和y的类型
X = np.matrix(X.values)
y = np.matrix(y.values)
#print(type(X))'
#'print(type(y))'
theta = np.matrix([[0],[0]])
print(X.shape,theta.shape,y.shape)
J = computeCost(X,y,theta)
print(J)


#使用sklearn预测模型
'''
model = lm.LinearRegression()
model.fit(X,y)
#矩阵.A等效于矩阵.getA() 把矩阵变成数组
x = np.array(X[:, 1].A1)
print(type(x),x.shape)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
'''


#===================gradient descent

def gradientDescent(X,y,theta,alpha,iteration):
        m = len(y)
        J_history = np.zeros(iteration)

        for i in range(iteration):
                #向量化
                theta = theta - alpha / m * X.T * (X * theta - y)
                J_history[i] = computeCost(X,y,theta)
        
        return theta, J_history
alpha = 0.01
iteration = 1500
final_theta , final_J = gradientDescent(X,y,theta,alpha,iteration)
print("computecost:",computeCost(X,y,final_theta),"\ntheta:",final_theta)
#绘制线性模型
#横坐标
x = np.linspace(data.Population.min(),data.Population.max(),100)
#纵坐标   取向量分量！！！不要把逗号写成冒号！！！
f = final_theta[0,0] + (final_theta[1,0] * x)
#axe.subplots() 生成子画布
fig, ax = plt.subplots(figsize=(6,4))
#生成线  保证 x，f的维度一致
ax.plot(x,f,'r',label = 'Prediction')
#生成散点图
ax.scatter(data['Population'],data['Profit'],label = 'Training Data')
#loc = 2 表示生成在左上角
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit')
plt.show()

# 绘制theta训练过程
fig, ax = plt.subplots(figsize = (8,4))
#np.arange() 返回等差数组
ax.plot(np.arange(0,iteration,1), final_J, 'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error Function')
plt.show()


#=================multivariable

path2 = 'E:\lessons\ML wu.n.g\machine-learning-ex1\machine-learning-ex1\ex1\ex1data2.txt'
data2 = pd.read_csv(path2, names=['Size','Bedrooms','Price'])
print(data2.head())

#特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

#预处理初始化
data2.insert(0, 'Ones', 1)
cols2 = data2.shape[1]
x2 = data2.iloc[:,0:cols2-1]
y2 = data2.iloc[:,cols2-1:]
#转换成matrix形式
x2 = np.matrix(x2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix([[0],[0],[0]])

#计算梯度下降
iteration2 = 1000
final_theta2, final_J2 = gradientDescent(x2,y2,theta2,alpha,iteration2)
print(final_theta2,final_J2)
#描绘训练过程
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(np.arange(iteration2),final_J2,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()