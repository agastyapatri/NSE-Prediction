"""
Class Defining the Recurrent Neural Network
"""
import torch
import torch.nn as nn
import numpy as np

class LSTMNetwork(nn.Module):
    """
    Class defining the LSTM network to predict stock prices
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
        network_LSTM = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),

            nn.LSTM()


        )
        return network_LSTM


    def testmethod(self):
        print("LSTMNetwork is imported")


if __name__ == "__main__":
    pass