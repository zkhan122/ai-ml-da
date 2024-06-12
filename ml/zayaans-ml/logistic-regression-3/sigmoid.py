# sigmoid is a activation function used to map input between 0 and 1 then classification can be done based on decision boundary
import numpy as np
import math

def sigmoid(input):
    x = input
    # without numpy
    #y = 1 / (1 + math.exp(-1 * x))
    
    # with numpy
    y =  1 / (1 + np.exp(-x))
    return y

inputs = list(np.arange(1, 10, 0.5))
outputs = [sigmoid(y) for y in inputs]

print("Inputs: ", inputs)

print("\nOutputs (Sigmoid applied): ", outputs)