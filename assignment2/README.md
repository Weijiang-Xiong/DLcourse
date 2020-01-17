### Assignment2

#### mnistLoader.py

`loadAll()` wraps all functions in the script and returns trainData, validationData, and testData for mnist dataset. 
`labelOneHot()` encode the original label into one hot labels
`loadMNIST()` is a little bit naive, and returns a 2d matrix that contains all the data, but this runs rather slow...so, it is not utilized in `loadAll()`...instead, a faster version `fastLoadMNIST()` is used.

Note that the whole loading process takes 5 minutes or so.. and once complete, the data will be dumped using pickle.

#### Networks.py

Generally, this script contains a class `FCNet()` which is a fully connected network with softmax output. 

You may choose to initialize a single layer network with `oneLayerNet = Networks.FCNet((784,10), 'sigmoid')` in which the tuple `(784,10) `means the number of neurons in each layer, and 'sigmoid' is the type of activation function (you may also choose 'relu'). The parameters are randomly initialized.

After loading data and initializing the network parameters, you can use `net.gradientDescent()` to train the network, and `net.report()` to test the model on test data.