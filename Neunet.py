import numpy as np
import matplotlib.pyplot as plt

class Neunet:
    # Neural Network object has a list of weights, bias, and 
    def __init__(self, l_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.lrate = l_rate

    # Limits output from 0 to 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of the sigmoid function for error calculations   
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Predicts output given a datapoint using double layer neural network
    def predict(self, iv):
        l1 = np.dot(iv, self.weights) + self.bias
        l2 = self.sigmoid(l1) 
        return l2

    # Calculates the derivative of the error and returns the change needed to biases and weights
    def error(self, iv, target):
        l1 = np.dot(iv, self.weights) + self.bias
        l2 = self.sigmoid(l1) 
        p = l2

        dpe = 2 * (p - target) #deriv prediction error
        dl1 = self.sigmoid_deriv(l1) #deriv prediction layer 1
        dl1b = 1 #deriv layer 1 bias
        dl1w = (0 * self.weights) + (1 * iv) #deriv layer 1 weights

        dbe = dpe * dl1 * dl1b #deriv bias error
        dwe = dpe * dl1 * dl1w #deriv weights error

        return dbe, dwe

    def update(self, dbe, dwe): # Updates the biases and weights based on the error
        self.bias = self.bias - (dbe * self.lrate)
        self.weights = self.weights - (dwe * self.lrate)

    def train(self, ivs, targets, iter): # Trains the model iter number of times on the dataset and the cumulative error for each iteration
        errors = []

        for i in range(iter):
            index = np.random.randint(len(ivs))

            input = ivs[index]
            target = targets[index]

            dbe, dwe = self.error(input, target)

            self.update(dbe, dwe)

            if i % 100 == 0:
                error = 0
                
                for x in range(len(ivs)):
                    iv = ivs[x]
                    t = targets[x]

                    p = self.predict(iv)
                    e = np.square(p - t)

                    error += e
                
                errors.append(error)
            
        return errors