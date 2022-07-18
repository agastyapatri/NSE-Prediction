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
        IBM = loader.alpha_vantage_data(outputsize="full")
        IBM_split = loader.split_data(IBM, target="Close", ratio=0.85, norm=True)

        return IBM_split


    def Define_Model(self):
        model = MultiLayerPerceptron(input_size=4, output_size=1, hidden_sizes=[3,2])
        network = model.network()
        return network


    def Train_Model(self):
        net = self.Define_Model()
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

    stockpred = StockPrediction(run = True, visualize = False)


    # Running the Model
    if stockpred.run == True :
        print("Running the Model")
        ibm = stockpred.Load_Data()
        train_data, train_target, test_data, test_target = ibm

        net = stockpred.Define_Model()




        # if stockpred.visualize == True:
        #     stockpred.Visualize_Data()
        #

        # stockpred.Train_Model()
        # stockpred.Evaluate_Model()






