# This site may be helpful  https://www.python-course.eu/neural_network_mnist.php
#%%
import numpy as np
from sklearn.utils import shuffle

#%%
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def derSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    z[z<0]=0
    return z

def derRelu(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z

def actFun(z, actType):
    funSet = {
        'sigmoid': sigmoid,
        'relu': relu
    }
    fun = funSet.get(actType, sigmoid)
    return fun(z)

def derFun(z, actType):
    derSet = {
        'sigmoid': sigmoid,
        'relu': relu
    }
    derFun = derSet.get(actType, derSigmoid)
    return derFun(z)

def softmax(weight, bias, a):
    # zs.shape = (m,n) where m is the number of neurons, 
    # and n is the number of samples.
    zs = np.dot(weight, a) + bias
    return np.exp(zs)/np.sum(np.exp(zs), axis = 0), zs
#%%
class FCNet():
    """class Net() creates a fully connected neural network with several layers, in which the last layer is softmax
    """
    def __init__(self, size:list, actType:str):
        self.numLayers = len(size)
        self.size = size
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        self.actType = actType  # 'sigmoid', 'relu'


    def forwardPass(self, x):
        """[forward pass calculate the output of the whole network]
        Arguments:
            x {[numpy vector]}
        Output: 
            actValue: the activation value of the final layer
            zs: the z value of each layer
            actValues: the activation value of each layer

        """
        actValue = x
        actValues = [x]
        zs = []
        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(weight, actValue)+bias
            zs.append(z)
            actValue = actFun(z,'sigmoid')
            actValues.append(actValue)
        actValue, zf = softmax(self.weights[-1], self.biases[-1], actValue)
        actValues.append(actValue)
        zs.append(zf)
        return actValue, zs, actValues

    def backwardPass(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # # feedforward
        # activation = x
        # activations = [x] # list to store all the activations, layer by layer
        # zs = [] # list to store all the z vectors, layer by layer
        # for b, w in zip(self.biases, self.weights):
        #     z = np.dot(w, activation)+b
        #     zs.append(z)
        #     activation = sigmoid(z)
        #     activations.append(activation)

        _, zs, activations = self.forwardPass(x.transpose())

        # backward pass
        delta = self.deltaCE(activations[-1], y.transpose())
        nabla_b[-1] = delta
        # take care for the shape of delta..
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = derFun(z, self.actType)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def gradientDescent(self, trainData, numEpoch, batchSize, eta, valData = None):
        """[gradientDescent() uses mini-batch gradient descent to train the network]
        """
        n = trainData[0].shape[0]
        if valData:
            nVal = valData[0].shape[0]

        for j in range(numEpoch):
            #np.random.shuffle(trainData)
            trainData[0], trainData[1] = shuffle(trainData[0], trainData[1])
            batches = [(trainData[0][k:k+batchSize], trainData[1][k:k+batchSize]) for k in range(0, n, batchSize)]
            for batch in batches:
                self.batchUpdate(batch, eta)
            if valData:
                print('The {0}th epoch completed'.format(j))
                print('On trainData, correctly classify {1}/{0} images.'.format(n, self.predict(trainData)))
                print('On trainData, correctly classify {1}/{0} images.'.format(nVal, self.predict(valData)))
            else:
                print('The {0}th epoch completed'.format(j))

    def batchUpdate(self, batch, eta):
        """use a batch to update weights and biases of the network
        Arguments:
            batch {list of tuples from zip() function} 
            -- [each tuple contains a img array and a label]
            eta {[scalar]} -- [learning rate]
        """
        # the sizes of the laysers are different..so how can i vectorize ??
        grad2b = [np.zeros(b.shape) for b in self.biases]
        grad2w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b , delta_nabla_w = self.backwardPass(batch[0], batch[1])
        grad2b = [nb+np.sum(dnb,axis=1).reshape(nb.shape) for nb, dnb in zip(grad2b , delta_nabla_b)]
        grad2w = [nw+dnw for nw, dnw in zip(grad2w , delta_nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases , grad2b)]
        self.weights = [w-eta*nw for w, nw in zip(self.weights , grad2w)]
        # for x, y in batch:
        #     delta_nabla_b , delta_nabla_w = self.backwardPass(x, y)
        #     grad2b = [nb+dnb for nb, dnb in zip(grad2b , delta_nabla_b)]
        #     grad2w = [nw+dnw for nw, dnw in zip(grad2w , delta_nabla_w)]
        #     self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases , grad2b)]
        #     self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights , grad2w)]

    def predict(self, testData):
        probs, _, _ = self.forwardPass(testData[0].transpose())
        result = np.zeros_like(probs)
        index = np.argmax(probs, axis=0)
        result[index, np.arange(probs.shape[1])] = 1
        correct = np.sum(result * testData[1].transpose())
        return correct

    def report(self, testData):
        print('\nOn test set, correctly classify {1}/{0} images.\n'.format(testData[1].shape[0], self.predict(testData)))

    def deltaCE(self, a, y):
        """Return the delta for the output layer when using cross 
        entropy loss."""
        return (a-y)


#%%
