import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#Scipy库中的optimiz可以实现matlab中fminunc的功能，来优化函数计算成本和梯度
import scipy.optimize as opt
#高级处理分类的一个信息report
from sklearn.metrics import classification_report as cr
#=================visualizing data

path = 'E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex2\ex2\ex2data1.txt'
data = pd.read_csv(path, names=['ex1score', 'ex2score', 'admitted'])
print(data.head(),data.describe())
#dataframe有布尔筛选功能 ，筛选出符合条件的数据
#isin可以进行布尔判断该属性的值是否符合isin的内容
positive = data[data.admitted.isin(['1'])]
negetive = data[data.admitted.isin(['0'])]
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(positive['ex1score'],positive['ex2score'], c='b', label = 'Admitted')
ax.scatter(negetive['ex1score'],negetive['ex2score'],c='r', s=50, marker='x', label = 'Failed')
#获取轴的位置
pos = ax.get_position()
#设置轴的新位置（left，bottom，width，heigeth）
ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.8])
#设置图例在图的上方外侧 bbox_to_anchor:表示legend位置，前一个左右后上下，使用后
#loc将不再起正常作用； ncol = 3 表示图例的几列显示 
#该句写法其实是原作者通过试探得来的参数。。。。
ax.legend(loc = 'center left', bbox_to_anchor=(0.3, 1.12), ncol = 3)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#plt.show()

#=================sigmoid function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
'''画出sigmoid函数
x1 = np.arange(-10, 10, 0.1)
plt.plot(x1, sigmoid(x1),c = 'r')
plt.show()
'''

#================cost function

def cost(theta , X, Y):
    first = (-Y) * np.log(sigmoid(X @ theta))
    second = (1 - Y) * np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)
#初始化数据
if  'Ones' not in data.columns:
     data.insert(0, 'Ones', 1)
#这里写法与ex1不同，此时将X和Y转换成了Numpy - array而不是matrix     
X = data.iloc[:, :-1].as_matrix()
Y = data.iloc[:,-1].as_matrix().reshape(-1)
theta = np.zeros(X.shape[1]).T
#print(cost(theta, X ,Y))

#==================Gradient descent

def gradient(theta, x, y):
     return (x.T @ (sigmoid(x @ theta) -y)) / len(y)
#print(gradient(theta, X, Y))
#使用高级优化函数optimize来运算速度远远超过梯度下降，只需传入cost函数、theta和梯度
#注意在使用时cost函数定义theta要放在第一个
#算法有很多种类这里选用fimin_tnc（Newton Conjugate-Gradient）
result = opt.fmin_tnc(func = cost, x0 =theta ,fprime = gradient, args=(X,Y))
#返回array解决方案；迭代次数；一个XX编码
#print(result)
#res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
#以上为另一种方法，minimize可以选择更多方法
#print("final_cost = ", cost(result[0], X, Y))


#==================Using model to predtict

def predict(theta, X):
     probability = sigmoid(X @ theta)
     return [1 if x >= 0.5 else 0 for x in probability]
final_theta = result[0]
#print('predictions=' ,predict(final_theta, X))
predictions = predict(final_theta, X)
#zip()函数，参数是迭代对象，作用将相同位置的元素打包成一个个的tuple
correct = [1 if a==b else 0 for (a,b) in zip(predictions, Y)]
accuracy = sum(correct) / len(Y)
#print(accuracy)
#也可用sklearn中的方法进行检查
#print(cr(predictions,Y))


#==================Decision boundary

#X＠theta = 0
x1 = np.arange(0, 130, step =0.1)
x2 = -(final_theta[0] + x1 * final_theta[1]) / final_theta[2]
#绘图
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(positive['ex1score'], positive['ex2score'], c = 'b', label = 'Admitted')
ax.scatter(negetive['ex1score'], negetive['ex2score'], c = 'r', s = 50, marker = 'x', label= 'Failed')
#画初决策边界
ax.plot(x1, x2)
pos = ax.get_position()
#设置轴的新位置（left，bottom，width，heigeth）
ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.8])
ax.legend(loc = 'center left', bbox_to_anchor=(0.3, 1.12), ncol = 3)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.set_title('Decision Boundary')
#plt.show()
