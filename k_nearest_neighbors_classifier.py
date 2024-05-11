#KNN mechanism of classification

import numpy as np
from sklearn import datasets

# Preparing Data
np.random.seed(0)
iris = datasets.load_iris()
x = iris.data
y = iris.target
i = np.random.permutation(len(iris.data))
x_train = x[i[:-10]]
y_train = y[i[:-10]]
x_test = x[i[-10:]]
y_test = y[i[-10:]]

#Apply KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto',
                     leaf_size = 30,
                     metric = 'minkowski',
                     metric_params=None,
                     n_neighbors=5,
                     p=2,
                     weights='uniform')

#Predict and compare
print(knn.predict(x_test))
print(y_test)
# We can see that we obtained a 10% error. index number 1 (second from left) is error compare to y_test

#Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x = iris.data[:,:2]
y = iris.target
x_min, x_max = x[:,0].min() -.5,x[:,0].max() + .5
y_min, y_max = x[:,1].min() -.5,x[:,1].max() + .5
#MESH
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA', '#FFAAAA'])
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max,h))
knn.fit(x,y)
Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light, shading='auto')
#Plot the training points
plt.scatter(x[:,0],x[:,1],c=y)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Visualization of Sepals Decision Boundaries")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
