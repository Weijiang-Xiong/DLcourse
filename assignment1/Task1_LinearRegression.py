"""
This file presents a procedure that trains a linear
regression model to predict the published relative performance (PRP)
of a CPU, using 8 features from the Computer Hardware Dataset.

Importing packages and loading data with pandas completes before
the data processing begins. 
Then comes data encoding, in which non-numerical
features are mapped into the non-negative integer space.
After that, all data are transferred into numpy.array and split into 
training set (70%) and testing set (30%) randomly.
A gradient descent method is developed to train a parameter vector 
that minimizes the MSE of the training set.
The latest 5 training errors are examined every 50 epoch, and learning
rate will be lowered when training error almost converges.
The trained model has a testing error at around 2643.
Besides, the algebraic solution for the regression task is also
calculated for comparison and its testing error is 2634.
"""
# %%
# import packages
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing as pre
from IPython.core.interactiveshell import InteractiveShell
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split

InteractiveShell.ast_node_interactivity = 'all'

# %%
# load data from the file

colNames = ['vendor', 'name', 'MYCT', 'MMIN',
            'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
data = pd.read_csv("data/cpu_performance/machine.data",
                   delimiter=',', header=0, names=colNames)
data.head()

# %%
# use labelencoder to facilitate computing
X, y = data.iloc[:, 0:8], data.iloc[:, -2]

# encode the 1st and 2nd columns before transforming them into
# numpy array, or the dataframe will be transformed into an 'object',
# which will results in an error when calculating std

# X, y = data.iloc[:, 2:8], data.iloc[:, -2]
X.iloc[:, 0] = pre.LabelEncoder().fit_transform(data.iloc[:, 0])
X.iloc[:, 1] = pre.LabelEncoder().fit_transform(data.iloc[:, 1])
X, y = np.array(X), np.array(y)
# x1 = pre.LabelEncoder().fit_transform(data.iloc[:, 0])
# x2 = pre.LabelEncoder().fit_transform(data.iloc[:, 1])
# X = np.concatenate((x1,x2,X), axis=1)

# %%

# split data into training and testing sets, and normalize all of them except yTest

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=10)
featMean, featStd = np.mean(XTrain, axis=0), np.std(XTrain, axis=0)
XTrain = (XTrain - featMean) / featStd
XTest = (XTest - featMean) / featStd

# %%

# initialize parameter
theta = np.zeros(XTrain.shape[1]+1)

# Concatenate X with a new dimension for bias
XTrain = np.concatenate((np.ones((XTrain.shape[0], 1)), XTrain), axis=1)
XTest = np.concatenate((np.ones((XTest.shape[0], 1)), XTest), axis=1)

# %% 

# compute hypothesis and loss
initRate = 0.0001
learningRate = initRate
# hyp = XTrain@theta
# MSE = 0.5/yTrain.shape[0]*sum((hyp-yTrain)**2)
# grad = (hyp-yTrain)@XTrain
# theta -= learningRate*grad

# train a linear model using SGD

numEpoch = 1000
lossList = []
for epoch in range(numEpoch):
    # forward
    hyp = XTrain@theta
    MSE = 0.5/yTrain.shape[0]*sum((hyp-yTrain)**2)
    grad = (hyp-yTrain)@XTrain
    theta -= learningRate*grad
    if epoch % 50 == 0:
        print('Epoch', epoch, 'loss:', MSE)
        lossList.append(MSE)
        if len(lossList) > 5:
            currentLoss = np.array(lossList[-5:-1])
            if currentLoss.std()/currentLoss.mean() < 0.01:
                learningRate *= 0.9
                print('almost converged, lowering learning rate')
        if learningRate/initRate < 0.5:
            print('solution converged, exit training')
            break


# %%
# calculate the testing error of theta and compare it with algebraic solution
hypTest = XTest@theta
MSETest = 0.5/yTest.shape[0]*sum((hypTest-yTest)**2)
print("test error using gradient descent: ", MSETest)

algSolu = np.linalg.inv((XTrain.transpose()@XTrain))@XTrain.transpose()@yTrain
algTest = XTest@algSolu
MSEalg = 0.5/yTest.shape[0]*sum((algTest-yTest)**2)
print("test error using linear algebra: ", MSEalg)
# %%
