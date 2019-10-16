# %%
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# fileName = 'assignment2/data/train.zip'
#%%
def loadMNIST(fileName:str):
    mnistData = np.array([])
    with ZipFile(fileName) as archive:
        # the first of namelist is empty....so...
        for entry in archive.namelist()[1:]:
            with archive.open(entry) as file:
                curImg = Image.open(file)
                curImgArr = np.array(curImg)
                curLabel = entry.rsplit(sep='.')[0].rsplit(sep='num')[-1]
                curRow = np.concatenate((curImgArr.reshape(curImgArr.size,), np.array([curLabel])), axis= 0)
            if mnistData.size ==0:
                mnistData = curRow.reshape(1,curRow.size)
            else:
                mnistData = np.concatenate((mnistData, curRow.reshape(1,curRow.size)), axis = 0)
            # print(mnistData.shape)
    return mnistData

def fastLoadMNIST(fileName:str):
    with ZipFile(fileName) as archive:
        k = 0     
        mnistData = np.empty((len(archive.namelist())-1,28,28))
        labels = np.empty((len(archive.namelist())-1,1), dtype=int)
        # mnistData = np.empty((100,28,28))
        # labels = np.empty((100,1), dtype=int)
        # the first of namelist is empty....so...
        for entry in archive.namelist()[1:]:
        # for entry in archive.namelist()[1:100]:
            with archive.open(entry) as file:
                curImg = Image.open(file)
                curImgArr = np.array(curImg)
                curLabel = entry.rsplit(sep='.')[0].rsplit(sep='num')[-1]
            mnistData[k,:,:] = curImgArr
            labels[k] = int(curLabel)
            # print(k)
            k += 1
    return mnistData, labels

def saveMNIST(data, name:str):
    data.dump(name)
    print('{0} is saved to file {}'.format(data, name))
    
def loadCompressedMNIST(path:str):
    mnistData = np.load(path)
    print('{0} has been loaded'.format(path))
    return mnistData

def labelOneHot(j):
    e = np.zeros((j.size,10))
    e[np.arange(j.size),j.reshape((1,j.size))] = 1.0
    return e

def loadAll(train:str, test:str):
    img, label = fastLoadMNIST(train)
    label = labelOneHot(label)
    testImg, testLabel = fastLoadMNIST(test)
    testLabel = labelOneHot(testLabel)
    # the first dimension of img and label must be the same..so mnistData[k,:,:] = curImgArr the first index is k
    trainImg, validationImg, trainLabel, validationLabel = train_test_split(img, label, test_size=0.3, random_state=5)
    trainData = (trainImg, trainLabel)
    validationData = (validationImg, validationLabel)
    testData = (testImg, testLabel)
    return trainData, validationData , testData