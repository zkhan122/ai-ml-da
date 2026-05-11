
from typing import List
import numpy as np
from layer import Layer


class Network():
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def get_layers(self):
        return self.layers
    
    def print_layers(self):
        print(self.get_layers())


if __name__ == "__main__":
    
    input_node = Node()
    
    n1_1 = Node() # first node in layer 1
    n2_1 = Node()
    n1_2 = Node()
    n2_2 = Node()
    n3_2 = Node()
    
    output_node = Node()

    layer1 = Layer(1, [n1_1, n2_1])
    layer2 = Layer(1, [n1_2, n2_2, n3_2])
    
    network = Network([layer1, layer2])