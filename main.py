"""
Main content  of the project
"""

# Global Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
import yfinance

# Custom Imports: transformations
from src.visualize_data import Visualizer
from src.unpack_data import Unpacking
from src.model import  Model
from src.normalizer import Normalize

# Custom Imports: models
from models.MLP import MultiLayerPerceptron



class StockPrediction(nn.Module):
    """
    Class that performs the actual prediction based on the different network architectures defined in other modules.
        1. num_epochs = number of epochs for which training will occur
        2. learning_rate = rate at which the weights and biases are updated
        3. batches = number of batches in each training pass
    """

    def __init__(self, num_epochs, learning_rate, batches, momentum):
        super().__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batches = batches
        self.momentum = momentum


    def normalize(self, data):
        """
        Function to normalize the data fed to it
        :param data: DataFrame  or Tensor or Array
        :return: normalized data
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 1:]
            mu = data.mean()


        # elif isinstance(data, np.ndarray):
        #     stock_data = torch.tensor(data)
        # else:
        #     stock_data = data.clone()
        #
        # mu = torch.mean(stock_data)


        return mu[1]





        #
        #     stock_data = torch.from_numpy(np.array(stock_data))
        #
        # else:
        #     stock_data = data[:,1:]
        #     stock_data = torch.from_numpy(np.array(stock_data))
        #
        # mu = torch.mean(stock_data)
        # sigma = torch.std(stock_data)
        #
        # scaled_data = (stock_data - mu) / sigma





    def training(self):
        pass




    def validation(self):
        pass




    def evaluation(self):
        pass







if __name__ == "__main__":
    """
    Runner Code. Defines all the data and the transformations made to it.
    
        1. Loading the data / Generating the training and validation datasets.
        2. Visualizing the data
        3. Normalizing and Preparing the data
        4. Defining the Model
        5. Training the Model
        6. Hyperparameter optimization
        7. Testing the data 
        
    """

    path = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    ticker = "RELIANCE"
    model  = StockPrediction(num_epochs=100, learning_rate=0.001, batches=16, momentum=0.9)


    # 1. LOADING THE DATA
    unpacker = Unpacking(PATH=path, ticker=ticker)
    reliance_data, beginning_date, ending_date, features = unpacker.load_data()
    rel_tensor, rel_train, rel_validation = unpacker.to_tensor(reliance_data, 0.85)


    # VISUALIZING THE DATA
    visualizer = Visualizer(stock_data=reliance_data, ticker="RELIANCE",
                            start_date="2020-01-01", end_date="2021-01-01")






    """----------------------------------------------TESTING THE IMPORTS---------------------------------------------"""
    normalizer = Normalize(data = pd.DataFrame(np.random.randn(5,6)))
    normalizer.fit_transform()







    # Implementing the different steps required to train the data.














