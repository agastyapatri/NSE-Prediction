"""
Class to define the model that will be used to predict. This file will be edited constantly
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Class to define the models that will be used/optimized
        1. input_size (int) = the size of the input layer.
        2. hidden_sizes (list) = a list of sizes of each of the hidden layers.
        3. output_sizes (int) = the size of the output layer.
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes



    def MLP(self):
        """
        Function to define the Multi Layer Perceptron.
        """
        model_MLP = nn.Sequential(
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
        return model_MLP


    def RNN(self):
        """
        Function to define the Recurrent Neural Network
        """
        pass

    def CNN_LSTM(self):
        """
        Function to implement the CNN-LSTM defined in the paper
        """
        pass


if __name__ == "__main__":
    """
    Testing if the functions are working.
    """

    test = Model(input_size=5, output_size=1, hidden_sizes=[4,3,2])




