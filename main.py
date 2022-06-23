"""
Main content  of the project
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
import yfinance

# Custom Imports
from src.visualize_data import Visualizer
from src.normalizer import Normalize
from src.unpack_data import Unpacking
from src.model import  Model

from models.CNN import *

class StockPrediction(nn.Module):

    def __init__(self, num_epochs, learning_rate, batches, momentum):
        super().__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batches = batches
        self.momentum = momentum

    def training(self):
        pass




    def validation(self):
        pass








if __name__ == "__main__":
    """
    Runner Code. Defines all the data and the transformations made to it.
    """


    path = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    ticker = "RELIANCE"


    stockpred = StockPrediction(num_epochs=100, learning_rate=0.001, batches=16, momentum=0.9)

    # Loading the Data
    unpacker = Unpacking(PATH=path, ticker=ticker)
    reliance_data, beginning_date, ending_date, features = unpacker.load_data()
    rel_tensor, rel_train, rel_validation = unpacker.to_tensor(reliance_data, 0.85)

    # Visualizing the Data
    visualizer = Visualizer(stock_data=reliance_data, ticker="RELIANCE",
                            start_date="2020-01-01", end_date="2021-01-01")
    # visualizer.visualize(option="transformed", features=["Open", "Close"])
    # visualizer.visualize(option="transformed", features=["High","Low"])
    visualizer.OHLC(option="transformed")


    # Implementing the different steps required to train the data.

    """
        1. Loading the data
        2. Visualizing the data
        3. Normalizing the data
        3. Training the data 
        4. Hyperparameter optimization
        5. Testing the data
        
    """














