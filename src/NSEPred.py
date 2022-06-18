"""
NSE Stock Market Prediction, by Agastya Patri

NASDA API key: 1t2bzSTzY75Wdv6T_y-g
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime as dt
from torch import nn
import pandas as pd
import os

from NSE_Prediction.src.visualize_data import Visualizer

style.use("ggplot")

"""---------------------------------------------------------------------------------------------------------------------
Steps Taken in the modeling of the market:
    1. Preliminary Operations: getting data, making tensors, etc
    2. Standardizing the data: Normaalization 
    3. Creating the Model 
    4. Training the Model
    5. Testing the Model
    6. Evaluation Metrics 
---------------------------------------------------------------------------------------------------------------------"""


class Model(nn.Module):
    """
    Defining the Model that will predict the price of the stock
    """

    def __init__(self, ratio, input_size, output_size, learning_rate, num_epochs, num_batches):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.ratio = ratio

    """---------------------------------------------------------------------------------------------------------------------
    1. Visualizing the Data, and Creating Tensors
    ---------------------------------------------------------------------------------------------------------------------"""

    def visualize_data(self, stock_data, ticker, feature, start_date, end_date):
        """
        Function to plot the data for the stock.
        ticker: The ticker of the stock. This is subject to change.
        feature: The column that is intended to be viewed.
        start, end = the time range in years for which the data is required

        """

        start, end = stock_data.index.searchsorted(start_date), stock_data.index.searchsorted(end_date)

        plt.title(f"{feature} data for the stock: {ticker}")
        stock_data[feature].iloc[start:end].plot(c="b")
        plt.xlabel("Time")
        plt.ylabel(f"{feature}")
        plt.show()

    def to_tensor(self, stock_data):
        """
        Function to convert the pandas DataFrame Object to pytoch Tensor obeject. Takes in a DataFrame
        :return: Tensors and arrays
        """
        train_indices = int(self.ratio * len(stock_data))

        train_data = stock_data.iloc[:train_indices, 1:]
        validation_data = stock_data.iloc[train_indices:, 1:]

        train_data = torch.tensor(np.array(train_data.values), dtype=torch.float32)
        validation_data = torch.tensor(np.array(validation_data.values), dtype=torch.float32)
        stock_tensor = torch.tensor(np.array(stock_data), dtype=torch.float32)

        return stock_tensor, train_data, validation_data

    def normalize(self, tensor):
        """
        Funtion to scale the data in a way that makes it palatable to the model.
        :return: Normalized data
        """
        pass

    """---------------------------------------------------------------------------------------------------------------------
    2. Defining the Model
    ---------------------------------------------------------------------------------------------------------------------"""

    def model(self, stock_data):
        """
        Function to define the network.
        """
        stock_tensor, train_tensor, validation_tensor = self.to_tensor(stock_data)
        train_tensor = self.normalize(train_tensor)

        print(torch.mean(train_tensor))


if __name__ == "__main__":
    """---------------------------------------------------------------------------------------------------------------------
    0. Getting Stock Price Data, and Preliminary Options
    ---------------------------------------------------------------------------------------------------------------------"""

    PATH = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    tickers = os.listdir(PATH)


    def unpack_data(path, ticker):
        """
        Function to return unpack the data from a local csv file

        :return: stock_data, stock_ticker, beginning_date, ending_date
        """
        stock = ticker + ".NS.csv"

        # Loading data from local store
        stock_data = pd.read_csv(os.path.join(path, stock), parse_dates=True)

        # turning removing the index
        stock_data.reset_index(inplace=True)

        # Setting the date as the index
        stock_data.set_index("Date", inplace=True)

        # features, minus the index
        features = stock_data.columns[1:]

        # getting the beginning dates of te stock.

        beginning_date = stock_data.index[0]
        ending_date = stock_data.index[-1]
        return stock_data, ticker, beginning_date, ending_date, features


    # RELIANCE stock price. THIS IS THE DATA TO BE FED INTO THE MODEL
    reliance, rel_ticker, beginning_date, ending_date, features = unpack_data(PATH, "RELIANCE")

    model1 = Model(ratio=0.9, input_size=len(features), output_size=1, learning_rate=None, num_epochs=None,
                   num_batches=None)
