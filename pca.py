from sklearn.decomposition import PCA
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Load Data
iris = datasets.load_iris()

# Make PCA Analysis
covar_matrix = PCA(n_components=4)  # Check for the 4 feature sepal len,width petal len,width
covar_matrix.fit(iris.data)
variance = covar_matrix.explained_variance_ratio_
var = np.cumsum(np.round(variance, decimals=3)*100)
print(var)

# Plot the score of PCA
plt.ylabel("% Variance explained")
plt.xlabel("# of Features")
plt.title("PCA Analysis")
plt.ylim(90, 100.5)
plt.xticks([0,1,2,3], [1,2,3,4])
plt.axvline(2, linestyle="--", c="#bbbbbb")
plt.plot(var)
plt.show()

# 3 features is enough to build good model

# Dimensional Reduction
x_reduced = PCA(n_components=3).fit_transform(iris.data)
