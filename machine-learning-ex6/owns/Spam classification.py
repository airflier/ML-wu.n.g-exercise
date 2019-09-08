#打开邮件样本
with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\emailSample1.txt'
          , 'r'
            ) as f:
    email = f.read()
    print(email)
#处理邮件的方法，常将url，email address ， dollars 等标准化，忽略其内容
'''
1. Lower-casing: 把整封邮件转化为小写。
  2. Stripping HTML: 移除所有HTML标签，只保留内容。
  3. Normalizing URLs: 将所有的URL替换为字符串 “httpaddr”.
  4. Normalizing Email Addresses: 所有的地址替换为 “emailaddr”
  5. Normalizing Dollars: 所有dollar符号($)替换为“dollar”.
  6. Normalizing Numbers: 所有数字替换为“number”
  7. Word Stemming(词干提取): 将所有单词还原为词源。例如，“discount”, “discounts”, “discounted” and “discounting”都替换为“discount”。
  8. Removal of non-words: 移除所有非文字类型，所有的空格(tabs, newlines, spaces)调整为一个空格.

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import pandas as pd
import re      #regular expresion for e-mail processing

#英文分词算法 Porter stemmer
#from stemming.porter2 import stem

#英文分词算法
import nltk, nltk.stem.porter

#process the email(Exclude word stemming and removal of non-words)
def processEmaill(email):   
    #小写
    emali = email.lower()
    #移除html标签，[^],我认为有加号
    email = re.sub('<[^<>]+>', ' ', email)
    #匹配//后面不是空白字符的内容，遇到空白字符则停止 \s空白字符
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # '+' 和 '*'的区别
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email

#====================stem and removal of non-words
def emali2TokenList(email):
    stemmer = nltk.stem.porter.PorterStemmer()
    email = processEmaill(email)
    #将邮件分为各个单词
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    #遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        #删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # Porter stem 提取词根
        stemmed = stemmer.stem(token)
        #去除空字符串''
        if not len(token):
            continue
        tokenlist.append(stemmed)
    return tokenlist
    
#===================Vocabulary List

def get_vocab_list():
    vocab_dict = {}
    with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\\vocab.txt') as f:
        for line in f:
            (val, key) = line.split()
            vocab_dict[int(val)] = key

    return vocab_dict
vocab = get_vocab_list()
print(vocab[9])

def email2VocabIndices(token, vocab):
    #提取存在单词索引
    #!!!!! 注意key的初始值 （报错key error）
    index = [i for i in range(1, 1899) if vocab[i] in token]
    return index


with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\spamSample1.txt', 'r') as emailpath:
    path = emailpath.read()

#rint('len(vocab)=',len(vocab))
processEmaill(path)
token = emali2TokenList(path) #list
#print(token)
indices = email2VocabIndices(path, vocab)
#print(indices)


#==========================extracting features from Emails
 
def email2FeatureVector(email, vocab):
    # 将email转化为词向量，n是vocab的长度。存在单词的相应位置的值置为1，其余为0
    df = pd.read_table('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\spamSample1.txt'
                        , names = ['words'])
    
    #change to form of array
    voc = df.as_matrix()
    print(voc)
    #init the vector
    vector = np.zeros(len(vocab))
    voc_indices = email2VocabIndices(email, vocab)
    #有单词的地方0置为1 
    for i in voc_indices:
        vector[i] = 1
    return vector

vector = email2FeatureVector(path, vocab)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))
print(vector)



#=======================traiining svm 

'''Training set'''
mat1 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\spamTrain.mat')
X, y = mat1['X'], mat1['y'] #x(4000, 1899), y(4000, 1)
#print(X.shape, y.shape)

''' Test set'''
mat2 = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\spamTest.mat')
Xtest, ytest = mat2['Xtest'], mat2['ytest']

clf = svm.SVC(C = 0.1, kernel = 'linear')
clf.fit(X,y.flatten())

predTrain = clf.score(X, y)
predTest = clf.score(Xtest, ytest)

print(predTest,predTrain)





