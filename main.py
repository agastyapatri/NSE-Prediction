"""
Implementing all the steps of predictive modeling
"""

# Global Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os

# Custom Imports: transformations
from src.unpack_data import Unpacking
from src.visualize_data import Visualizer

# Custom Imports: models
from models.MLP import MultiLayerPerceptron
from models.CNN import ConvolutionalNetwork

# Custom Imports: Training + Testing
from src.trainer import Train



class StockPrediction(nn.Module):
    """
    Class that performs the actual prediction based on the different network architectures defined in other modules.
        1. num_epochs = number of epochs for which training will occur
        2. learning_rate = rate at which the weights and biases are updated
        3. batches = number of batches in each training pass
    """

    def __init__(self, run, visualize):
        super(StockPrediction, self).__init__()
        self.run = run
        self.visualize = visualize


    def Load_Data(self):
        # getting data from alpha vantage / local CSV files

        loader = Unpacking(PATH=None, ticker="IBM")
        stockdata = loader.alpha_vantage_data(outputsize="full")
        stockdata = loader.load_data()
        stock_split = loader.split_data(stock_tensor=stockdata, target_feature="Close", ratio=0.85)

        normed_IBM_data = loader.normalize_data(stockdata)



        return normed_IBM_data


    def Define_Model(self):
        pass

    def Train_Model(self):
        pass

    def Evaluate_Model(self):
        return None







if __name__ == "__main__":
    """
    1. Installing Python Dependencies
    2. Data Preparation: acquiring financial market data from Alpha Vantage
    3. Data Preparation: normalizing raw data
    4. Defining the LSTM model 
    5. Model Training
    6. Model Evaluation 
    7. Prediction future stock prices. 
    """

    stockpred = StockPrediction(run = True, visualize = None)


    # Running the Model
    if stockpred.run == True :
        print("Running the Model")
        ibm = stockpred.Load_Data()
        #
        # if stockpred.visualize == True:
        #     stockpred.Visualize_Data()
        #
        # stockpred.Define_Model()
        # stockpred.Train_Model()
        # stockpred.Evaluate_Model()






