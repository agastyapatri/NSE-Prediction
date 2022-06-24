"""
Class defining the Deep Feed Forward Network
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MultiLayerPerceptron(nn.Module):
    """
    Class to define the Feed Forward Network. The results from this network will be the baseline
    performance that will be improved with other architectures.

        1. input_size (int) = the size of the input layer.
        2. output_sizes (int) = the size of the output layer.
        3. hidden_sizes (list) = a list of sizes of each of the hidden layers.
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes



    def network(self):
        """
        Function to define the Feed Forward Network
        """
        network_MLP = nn.Sequential(
            # Input Layer
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),

            # Hidden Layer 1
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),

            # Hidden Layer 2
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            nn.ReLU(),

            # Hidden Layer 3
            nn.Linear(self.hidden_sizes[2], self.output_size)
        )
        return network_MLP


    def testmethod(self):
        print("MultiLayerPerceptron is imported")



if __name__ == "__main__":
    test_FFN = MultiLayerPerceptron(input_size=7, output_size=1, hidden_sizes=[5, 4, 3])
    test_data = torch.randn(10, 7)

    test_FFN.testmethod()
    net = test_FFN.network()