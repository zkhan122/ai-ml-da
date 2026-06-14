
from typing import List
import numpy as np
from layer import Layer
from utils import *


class Network():
    def __init__(self, a_in: np.array, W1, b1, W2, b2, W3, b3, iterations, alpha):
        self.a_in = a_in
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.iterations = iterations
        self.alpha = alpha

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
        #a_out_3 = np.expand_dims(a_out_3, axis=0)

        cache = {
            "A1": np.array(a_out_1),
            "A2": np.array(a_out_2),
            "A3": np.array(a_out_3)
        }

        print("Activated output:", a_out_3.shape)

        return a_out_3.T, cache

    
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
    

    def backprop_l2(self, dC_dA2, A1, A2, W2):
        
        dA2_dz2 = A2 * (1 - A2) # coming from layer after (backprop) -> this is the derivative of the sigmoid acti.vation applied on layer 2
        dC_dz2 = dC_dA2 * dA2_dz2 # this chains the gradient from the output layer (dC_dA2) to the sigmoid activation derivative through this hidden layer
        
        dz2_dw2 = A1
        dC_dw2 = np.dot(dC_dz2, dz2_dw2.T)

        dz2_db2 = 1

        # bias in layer 3 
        dC_db2 = np.sum(dC_dz2) 

        dz2_dA1 = W2
        dC_dA1 = np.dot(W2.T, dC_dz2)

        return dC_dw2, dC_db2, dC_dA1
    
    def backprop_l1(self, dC_dA1, A0, A1, W1): # W1 is not needed as it isnt used to backprop to input layer A0 cuz backprop is not applied to input layer

        dA1_dz1 = A1 * (1 - A1)
        dC_dz1 = dC_dA1 * dA1_dz1

        dz1_dw1 = A0
        dC_dw1 = np.dot(dC_dz1, dz1_dw1.T)
        dz1_db1 = 1
        dC_db1 = np.sum(dC_dz1)

        # dz1_dA0 = W1
        # dC_dA0 = np.dot(W1.T, dC_dz1)  # not needed as A0 is just the input layer and backprop is not applied to the input layer (again mentioned)

        return dC_dw1, dC_db1 # , dC_dA0
    
    def gradient_descent(self, dC_dw3, dC_db3, dC_dw2, dC_db2, dC_dw1, dC_db1):

        # w = w - alpha * dC/dw
        # b = b - alpha * dC/db 

        self.W3 = self.W3 - (self.alpha * dC_dw3)
        self.b2 = self.b2 - (self.alpha * dC_db3)

        self.W2 = self.W2 - (self.alpha * dC_dw2)
        self.b2 = self.b2 - (self.alpha * dC_db2)
        
        self.W1 = self.W1 - (self.alpha * dC_dw1)
        self.b1 = self.b1 - (self.alpha * dC_db1)

        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def train(self, Y):
        costs = []
        for epoch in range(self.iterations):
            y_hat, cache = self.forward_prop()

            error = cost(y_hat, Y)
            costs.append(error)

            # backprop
            dC_dw3, dC_db3, dC_dA2 = self.backprop_l3(y_hat, Y, y_hat.shape[0], cache["A2"], self.W3)
            dC_dw2, dC_db2, dC_dA1 = self.backprop_l2(dC_dA2, cache["A1"], cache["A2"], self.W2)
            dC_dw1, dC_db1 = self.backprop_l1(dC_dA1, cache["A0"], cache["A1"], self.W1)

            self.gradient_descent(dC_dw3, dC_db3, dC_dw2, dC_db2, dC_dw1, dC_db1)

            if epoch % 10 == 0:
                print(f"epoch {epoch+1}: cost = {error:4f}")
        
        return costs

        




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

    iterations = 100
    alpha = 0.1

    network = Network(np.array(X), W1, b1, W2, b2, W3, b3, iterations, alpha)
    # print(network.init_params())
    print(network.print_params())


    a_out = network.forward_prop()
    print("a_out", a_out)

    costs = network.train(y)