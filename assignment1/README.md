### Assignment1

```
# Taks1_LinearRegression

This file presents a procedure that trains a linear
regression model to predict the published relative performance (PRP)
of a CPU, using 8 features from the Computer Hardware Dataset.

Importing packages and loading data with pandas completes before
the data processing begins. 
Then comes data encoding, in which non-numerical
features are mapped into the non-negative integer space.
After that, all data are transfered into numpy.array and split into 
training set (70%) and testing set (30%) randomly.
A gradient descent method is developed to train a parameter vector 
that minimizes the MSE of the training set.
The latest 5 training errors are examed every 50 epoch, and learning
rate will be lowered when training error almost converges.
The trained model has a testing error at around 2643.
Besides, the algebraic solution for the regression task is also
calculated for comparison and its testin error is 2634.
```

```
#Task2_LogisticRegression

This file creates a multi-class classifier for the Glass identification 
dataset using logistic regression and gradient descent.

A multi-class classifer can be decomposed into multiple binary classifer,
so the codes are just wrapped like that.
This task seems quite difficult, because the scatter plot of the data points
are highly tangled...
The overall accuracy is 66.154%

The low accuracy may reslut from inadequate or improper features, which leads to these discussions:

Class 1~4 involves the jargon "float process", which is a technique to produce flat glass.
In float process, raw materials of glass are melt down and poured onto liquid stannum (Sn), 
and a dividing boundary forms between the material and Sn, just like oil and water. We can get 
a piece of flat glass though it. If we know the proportion of Sn in the glass, the model may be 
more confident to classify the two process techniques.

Reflective index (RI) belongs to optical features, but we may also want to know the clearness 
and color of each type of glass, since they are more closely connected with the usage. 
For example, extra clean glass usually contains little Fe.

The usage of glass depends on both material and process, it will be better if we can know more 
about the process of those glass, e.g., tempered, surface enhancement and poly-organic-materials. 
But that seems to be cheating, since the task is to classify the progress based on element composition.
```

### 