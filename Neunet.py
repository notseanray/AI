import numpy as np

class Neunet:
    def __init__(self, l_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.lrate = l_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, iv):
        l1 = np.dot(iv, self.weights) + self.bias
        l2 = self.sigmoid(l1) 
        return l2
        
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

    def update(self, dbe, dwe):
        self.bias = self.bias - (dbe * self.lrate)
        self.weights = self.weights - (dwe * self.lrate)