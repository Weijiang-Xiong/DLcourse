# import some library we are gonna use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap

def show_data_in_2d(X, y, title='dataset distribution of sepal length and sepal width',
                   xlabel='Sepal length', ylabel='Sepal width',
                   xidc=0, yidc=1):
    x_min, x_max = X[:, xidc].min() - .5, X[:, xidc].max() + .5
    y_min, y_max = X[:, yidc].min() - .5, X[:, yidc].max() + .5

    plt.figure(figsize=(10, 8))
    plt.title(title)

    # Plot the training points
    plt.scatter(X[:, xidc], X[:, yidc], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def plot_decision_regions(X, y, theta, resolution=0.02):

    plt.figure(figsize=(10, 8))
    # setup marker generator and color map
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = 1 / (1 + np.exp(-np.dot((np.array([xx1.ravel(), xx2.ravel()]).T), theta)))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
