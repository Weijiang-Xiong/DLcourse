# %%
import sys, os
sys.path.append(os.getcwd())
import assignment2.Networks as Networks
import assignment2.mnistLoader as mnistLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt 

trainFile = 'data/mnist/train.zip'
testFile = 'data/mnist/test.zip'

# %%
try:
    with open('data/mnist/allData.pkl', 'rb') as allData:
        trainData, validationData, testData = pickle.load(allData)
except:
    trainData, validationData, testData = mnistLoader.loadAll(trainFile, testFile)
    with open('data/mnist/allData.pkl', 'wb') as allData:
        pickle.dump((trainData, validationData, testData), allData)
        print('all data have been dumped into file')

# %%
trainData, validationData, testData = list(trainData), list(validationData), list(testData)
# plt.imshow(trainData[0][1846], cmap='gray')
# plt.title(trainData[1][1846])
# plt.show()
trainData[0] = trainData[0].reshape((-1,784))/255
validationData[0] = validationData[0].reshape((-1,784))/255
testData[0] = testData[0].reshape((-1,784))/255

#%%
print('\n Begin training a single layer net')
oneLayerNet = Networks.FCNet((784,10), 'sigmoid')
oneLayerNet.gradientDescent(trainData, 30, 11, 0.1, valData=validationData)
oneLayerNet.report(testData)

#%%
print('\n Begin training a double layer net')
twoLayerNet = Networks.FCNet((784,30,10), 'sigmoid')
twoLayerNet.gradientDescent(trainData, 30, 11, 0.1, valData=validationData)
twoLayerNet.report(testData)

#%%
