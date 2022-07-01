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
from src.visualize_data import Visualizer
from src.unpack_data import Unpacking
from src.model import Model
from src.configs import Configure

# Custom Imports: models
from models.MLP import MultiLayerPerceptron
from models.CNN import ConvolutionalNetwork



class StockPrediction(nn.Module):
    """
    Class that performs the actual prediction based on the different network architectures defined in other modules.
        1. num_epochs = number of epochs for which training will occur
        2. learning_rate = rate at which the weights and biases are updated
        3. batches = number of batches in each training pass
    """

    def __init__(self, num_epochs, learning_rate, num_batches, momentum):
        super().__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batches = num_batches
        self.momentum = momentum
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss()

    def train_model(self, training_data, training_labels, net):
        """
        Function to train the model in question
        :param trainind_data: a torch tensor.
        :return: trained model
        """
        training_loss = []
        train_accuracy_array = []
        optimizer = torch.optim.SGD(net.parameters(), lr = self.learning_rate)


        # Looping over all the epochs
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_count = 0
            incorrect_count = 0

            # resetting the gradient
            optimizer.zero_grad()
            predictions = net(training_data)[:,0]

            with torch.autograd.set_detect_anomaly(True):
                pass



            break

        pass














    def validation(self):
        pass




    def evaluation(self):
        pass







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


    """-----------------------------------------------------------------------------------------------------------------
    1. Loading the Data
    -----------------------------------------------------------------------------------------------------------------"""
    path = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    ticker = "RELIANCE"




    unpacker = Unpacking(PATH=path, ticker=ticker)
    total_reliance_data, beginning_date, ending_date, features = unpacker.load_data()
    normed_reliance_data = unpacker.normalize_data(total_reliance_data)



    """-----------------------------------------------------------------------------------------------------------------
    2. Normalizing and Preparing the Data
    -----------------------------------------------------------------------------------------------------------------"""

    total_data = unpacker.split_data(normed_reliance_data, ratio=0.90, target_feature="Close")
    normed_total_data = [ unpacker.normalize_data(dataobject) for dataobject in total_data ]

    # CONVERTING TO TORCH TENSORS
    train_data, validation_data, train_target, validation_target = [ unpacker.to_tensor(dataobject).float()
                                                                     for dataobject in normed_total_data]



    # VISUALIZING THE DATA
    visualizer = Visualizer(stock_data=total_reliance_data, ticker="RELIANCE", start_date=beginning_date,
                            end_date=ending_date)



    """-----------------------------------------------------------------------------------------------------------------
    3. Defining the Network
    -----------------------------------------------------------------------------------------------------------------"""

    # Defining the Multi Layer Perceptron (MLP)
    MLP = MultiLayerPerceptron(input_size=5, output_size=1, hidden_sizes=[5,4,3])
    MLP_NET = MLP.network()

    """-----------------------------------------------------------------------------------------------------------------
    4. Implementing the training process 
    -----------------------------------------------------------------------------------------------------------------"""

    MODEL  = StockPrediction(num_epochs=10, learning_rate=0.001, num_batches=None, momentum=0.9)


    # runmodel is a temporary function that will be called to test the train_model step
    runmodel = lambda a : MODEL.train_model(training_data=train_data, training_labels=train_target, net=MLP_NET)






