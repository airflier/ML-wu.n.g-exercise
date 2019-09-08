import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 

mat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex7\ex7\ex7data1.mat')
#print(mat.keys())
#print(mat)
X = mat['X']    #(50, 2)
#print(X.shape)
plt.figure(figsize=(8,8))
#facecolor is pots' color
plt.scatter(X[:,0],X[:,1], facecolors = 'none', edgecolors = 'r')
#plt.show()

#====================implementing pca

def featureNormalize(X):
    means = X.mean(axis = 0)
    #std ddof = 1 标准差是除以n - 1（无偏估计）
    stds = X.std(axis = 0, ddof = 1)
    X_norm = (X - means) / stds
    return X_norm, means, stds

def pca(X):
    sigma = (X.T @ X) / len(X)
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

X_norm, means, stds = featureNormalize(X)
U, S, V = pca(X_norm)
print(X_norm.shape)
#change to 1d , get the 1st column
print(U.shape,S.shape,V.shape)
print(U[:,0])
plt.scatter(X[:,0], X[:,1], facecolors='none', edgecolors='b')

plt.plot([means[0], means[0] + 5*S[0]*U[0,0]], 
         [means[1], means[1] + 5*S[0]*U[0,1]],
        c='r', linewidth=3, label='First Principal Component')
plt.plot([means[0], means[0] + 1.5*S[1]*U[1,0]], 
         [means[1], means[1] + 1.5*S[1]*U[1,1]],
        c='g', linewidth=3, label='Second Principal Component')
plt.grid(True)
plt.legend()
#plt.show()

#======================Dimensionality Reduction with PCA

#======projectData
def projectData(X, U, K):
    Z = X @ U[:,:K]
    return Z

Z = projectData(X_norm, U, 1) #Z[0] 1.481
#print(Z)

#======Reconstructing an approximaion of the data(重现数据维度)

def recData(Z, U, K):
    X_rev = Z @ U[:,:K].T
    return X_rev

X_rec = recData(Z, U, 1) #rec[0] = [-1.04741,-1.04741...]
#print(X_rec)

#======Visualizing

plt.figure(figsize=(8,8))
#x,y比例相同？？
plt.axis("equal")
plt.scatter(X_norm[:,0], X_norm[:,1], s = 50, facecolors = 'none',
            edgecolors='b', label = 'Original Data')
plt.scatter(X_rec[:,0], X_rec[:,1], s=30, facecolors='none', 
            edgecolors='r',label='PCA Reduced Data Points')
plt.title("reduced dimension show", fontsize = 20)
plt.xlabel("x1(Normalized)", fontsize = 14)
plt.ylabel("x2(Normalized)", fontsize = 14)
plt.grid(True)

#给对应的点连线
for x in range(X_norm.shape[0]):
    plt.plot([X_norm[x,0],X_rec[x,0]],[X_norm[x,1],X_rec[x,1]], 'k--')
plt.legend()
#plt.show()

#===========================Face Image Dataset
'''
Run PCA in face images
'''
facemat = loadmat('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex7\ex7\ex7faces.mat')
#print(facemat.keys())
X_face = facemat['X']   #(5000, 1024)
#print(X_face.shape)

#show the data
def displayData(X, row, col):
    fig, axs = plt.subplots(row, col, figsize=(10,10))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(X[r * col + c].reshape(32, 32).T, cmap = 'Greys_r')
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])

displayData(X_face, 10, 10)

#PCA
X_face_norm, face_means, face_stds = featureNormalize(X_face)

U_face, S_face, V_face = pca(X_face_norm) #U(1024,1024)  S(1024,)
print(U_face.shape, S_face.shape, V_face.shape)

displayData(X_face_norm, 6, 6)
#plt.show()

'U 的意义？？？？？？？？'
displayData(U_face[:,:100].T,10,10)
#plt.show()

#==================Dimension Redcution

z_face = projectData(X_face_norm, U_face, K = 36)
X_face_rec = recData(z_face, U_face, K = 36)

displayData(X_face_rec, 10, 10)
plt.show()