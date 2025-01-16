# sigmoid is a activation function used to map input between 0 and 1 then classification can be done based on decision boundary
import numpy as np
import matplotlib.pyplot as plt
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

print("\n Applied to distributed array: ")
z_tmp = np.arange(-10, 11)
out_tmp = sigmoid(z_tmp)
print(out_tmp)

np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, out_tmp])

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(z_tmp, out_tmp, c="b")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
ax.axvline(x=0, color='r', linestyle='--', label=f'Threshold = {0}')
ax.legend()
plt.show()