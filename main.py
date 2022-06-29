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
from src.model import  Model

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
        self.criterion = nn.CrossEntropyLoss


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



            optimizer.zero_grad()
            prediction = net(training_data)

            # Training Loss + Backprop
            with torch.autograd.set_detect_anomaly(True):

                # finding loss
                loss = self.criterion(prediction, training_labels)

                # backpropagation
                loss.backward(retain_graph = True)

                # updating weights
                optimizer.step()

            print(loss)







    def validation(self):
        pass




    def evaluation(self):
        pass







if __name__ == "__main__":

        
    """-----------------------------------------------------------------------------------------------------------------
    1. Preliminary Operations
    -----------------------------------------------------------------------------------------------------------------"""


    path = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    ticker = "RELIANCE"

    CNN = ConvolutionalNetwork()

    # The Prediction Model
    MODEL  = StockPrediction(num_epochs=100, learning_rate=0.001, num_batches=None, momentum=0.9)




    """-----------------------------------------------------------------------------------------------------------------
    2. Loading the Data
    -----------------------------------------------------------------------------------------------------------------"""

    unpacker = Unpacking(PATH=path, ticker=ticker)
    total_reliance_data, beginning_date, ending_date, features = unpacker.load_data()
    normed_reliance_data = unpacker.normalize_data(total_reliance_data)



    """-----------------------------------------------------------------------------------------------------------------
    3. Normalizing and Preparing the Data
    -----------------------------------------------------------------------------------------------------------------"""

    total_data = unpacker.split_data(normed_reliance_data, ratio=0.90, target_feature="Close")
    normed_total_data = [ unpacker.normalize_data(dataobject) for dataobject in total_data ]

    # 3. CONVERTING TO TORCH TENSORS
    train_data, validation_data, train_target, validation_target = [ unpacker.to_tensor(dataobject)
                                                                     for dataobject in normed_total_data]



    # 4. VISUALIZING THE DATA
    visualizer = Visualizer(stock_data=total_reliance_data, ticker="RELIANCE",
                            start_date=beginning_date, end_date=ending_date)



    """-----------------------------------------------------------------------------------------------------------------
    5. Defining the Network
    -----------------------------------------------------------------------------------------------------------------"""

    # Defining the Multi Layer Perceptron
    MLP = MultiLayerPerceptron(input_size=5, output_size=1, hidden_sizes=[5,4,3])
    net = MLP.network()
    MLP.testmethod()








    """----------------------------------------------TESTING THE IMPORTS---------------------------------------------"""









