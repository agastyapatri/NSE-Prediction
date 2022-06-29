"""
Class defining the Convolutional Neural Network
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class ConvolutionalNetwork(nn.Module):
    """
    Class to define the Convolutional Neural Network
    """

    def __init__(self):
        super().__init__()

    def testmethod(self):
        """
        Function to test if the module import is successful
        """

        print("ConvolutionalNetwork is imported")


if __name__ == "__main__":
    convnet = ConvolutionalNetwork()