from sklearn import datasets
iris = datasets.load_iris()
#print(iris.data)
#print(iris.target)
#print(iris.target_names)

import matplotlib.pyplot as plt

x = iris.data[:,2] # X-axis - Petal length
y = iris.data[:,3] # Y-axis - Petal width
species = iris.target # Species
x_min, x_max = x.min() - .5, x.max() + .5
y_min, y_max = y.min() - .5, y.max() + .5

#SCATTERPLOT
plt.figure()
plt.title("Iris Dataset - Classification By Petal Sizes")
plt.scatter(x,y, c=species)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()