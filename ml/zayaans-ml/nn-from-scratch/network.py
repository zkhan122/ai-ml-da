
from typing import List
import numpy as np
from layer import Layer
from utils import *


class Network():
    def __init__(self, a_in: np.array, W1, b1, W2, b2, W3, b3):
        self.a_in = a_in
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

    # def init_params(self):
    #     W1 = np.random.randn(3, 3)
    #     b1 = np.random.randn(3, 1) # zᴸ = Wᴸ * Aᴸ⁻¹ + bᴸ -> (3, 2)*(2, 1) = (3, 1)
    #     W2 = np.random.randn(3, 3) # batchsize=col=3 -> 3 features, 3 training samples
    #     b2 = np.random.randn(3, 1) # one bias term applied for every node (3 features 1 sample overall -> batch size = 1)
    #     W3 = np.random.randn(3, 1)
    #     b3 = np.random.rand(1, 1)

    #     return (W1, b1, W2, b2, W3, b3)
    
    def print_params(self):
        print("W1", self.W1)
        print("b1", self.b1)
        print("W2", self.W2)
        print("b2", self.b2)
        print("W3", self.W3)
        print("b3", self.b3)

    def forward_prop(self): # feed forward
        units = self.W1.shape[1] # batch size = m = num training samples
        a_out_1 = np.zeros(units)
        a_out_2 = np.zeros(units)
        a_out_3 = np.zeros(units)

        z_1 = np.dot(self.W1, self.a_in.T) + self.b1
        a_out_1 =  sigmoid(z_1)

        # w_2 = W2[:, j]
        z_2 = np.dot(self.W2, a_out_1) + self.b2
        a_out_2 = sigmoid(  z_2)

        # w_3 = W3[:, j]
        z_3 = np.dot(self.W3.T, a_out_2) + self.b3
        a_out_3 = sigmoid(z_3)
        return a_out_3.T

    
    def backprop_l3(self, y_hat, Y, m, A2, W3):
        A3 = y_hat
        
        dC_dz3 = (1 / m) * (A3 - Y) # dC_dZ3 = dC_dAL * dAL_dZL
        dz3_dw3 = A2
        dC_dw3 = np.dot(dC_dz3, dz3_dw3.T)

        dz3_db3 = 1

        # bias in layer 3 
        dC_db3 = np.sum(dC_dz3) # dz3_db3 is 1 anyways because dC_db3 = dC_dz3 * dC_db3

        dz3_dA2 = W3
        dC_dA2 = np.dot(W3.T, dC_dz3)

        return dC_dw3, dC_db3, dC_dA2
    

    def backprop_l2(self, y_hat, Y, m, A1, W2):
        A2 = y_hat
        
        dC_dz2 = (1 / m) * (A2 - Y) # dC_dZ3 = dC_dAL * dAL_dZL
        dz2_dw2 = A1
        dC_dw2 = np.dot(dC_dz2, dz2_dw2.T)

        dz2_db2 = 1

        # bias in layer 3 
        dC_db2 = np.sum(dC_dz2) # dz3_db3 is 1 anyways because dC_db3 = dC_dz3 * dC_db3

        dz2_dA1 = W2
        dC_dA1 = np.dot(W2.T, dC_dz2)

        return dC_dw2, dC_db2, dC_dA1
        

if __name__ == "__main__":

    X, y = feature_conv()
    print("feature and output shapes:")
    print(X.shape) # (4094, 3) -> 4094 rows, 3 cols for the features
    print(y.shape) # (4904, 1) -> 4094 rows, 1 col for the y_true

    W1 = np.random.randn(3, 3)
    b1 = np.random.randn(3, 1) # zᴸ = Wᴸ * Aᴸ⁻¹ + bᴸ -> (3, 2)*(2, 1) = (3, 1)
    W2 = np.random.randn(3, 3) # batchsize=col=3 -> 3 features, 3 training samples
    b2 = np.random.randn(3, 1) # one bias term applied for every node (3 features 1 sample overall -> batch size = 1)
    W3 = np.random.randn(3, 1) #  output node weight
    b3 = np.random.rand(1, 1) # output node bias

# input -> (4094, 3)
# W1 -> (3, 3), b1 -> (3, 1)
# z1 = (3, 3) * (4094, 3)^T + (3, 1) = (3, 4094) 


# W2 -> (3, 3), b2 -> (3, 1)
# z2 = (3, 3) * (3, 4094) + (3, 1) = (3, 4094)

# W3 -> (3, 1), b3 -> (1, 1)
# z3 = (3, 1)^T * (3, 4094) + (1, 1) = (1, 4094)^T = (4094, 1) -> y_pred (4094 rows, 1 col)


    network = Network(np.array(X), W1, b1, W2, b2, W3, b3)
    # print(network.init_params())
    print(network.print_params())


    a_out = network.forward_prop()
    print("a_out", a_out)

    costs = cost(a_out, y)
    print("Cost", costs)