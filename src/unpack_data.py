"""
Class to enable the visualization of data. This module also enables the use of the Alpha Vantage Stock
Price API.
"""

import numpy as np
import torch
from torch import nn
import pandas as pd
import os
from torch.nn import functional
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries


class Unpacking(nn.Module):
    """
    Provides the following methods:
        1. load_data()
        2. to_tensor()
        3. split_data()

        Although to_tensor() also splits the data, split_data() is a method present for redundancy.
    """

    def __init__(self, PATH, ticker):
        super().__init__()

        self.PATH = PATH
        self.ticker = ticker


    def load_data(self):
        """
        Function to take in csv data and cast it into usable form
        :input: path: the location of the data / ticker: the ticker of the asset to be loaded
        :return: stock_data (DataFrame), beginning_date (YYYY-MM-DD), ending_date(YYYY-MM-DD), features
        """

        stock = self.ticker + ".NS.csv"

        # Loading data from local store
        stock_data = pd.read_csv(os.path.join(self.PATH, stock))

        # Resetting the index
        stock_data.reset_index(inplace = True)
        stock_data.set_index("Date", inplace = True)

        # Getting the features of the data
        features = stock_data.columns[1:]

        # Getting the beginning and the ending dates of the stockdata
        beginning_date = stock_data.index[0]
        ending_date = stock_data.index[-1]

        return stock_data, beginning_date, ending_date, features



    def split_data(self, stock_data, target_feature, ratio):
        """
        Function to yield train and validation data from the input. Also returns the target variable
        :input: ratio: the ratio of train data to total data
        :return: train_data, validation_data, train_target, validation_target
        """
        train_indices = int(ratio * len(stock_data))

        target = stock_data[target_feature]
        stock_data = stock_data.drop(target_feature, axis=1)


        train_data = stock_data.iloc[:train_indices, 1:]
        validation_data = stock_data.iloc[train_indices:, 1:]

        train_target = target[:train_indices]
        validation_target = target[train_indices:]



        return train_data, validation_data, train_target, validation_target




    def to_tensor(self, dataframe):
        """
        Function to convert the pandas DataFrame Object to pytoch Tensor object
        :return: Tensors and arrays
        """
        tensor = torch.tensor(dataframe.values, requires_grad=True)
        return tensor



    def normalize_data(self, dataframe):
        """
        Function to normalize the data fed into it
        :param data: torch tensors
        :return:
        """
        normed_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
        return normed_dataframe



    def testmethod(self):
        print("unpack_data is imported")




if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------------------------------
    Testing the methods of the class, and the relations between the tensors
    -----------------------------------------------------------------------------------------------------------------"""

    unpacker = Unpacking(PATH ="/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/",
                         ticker = "RELIANCE")

    # LOADING THE DATA
    total_reliance_data = unpacker.load_data()
    reliance_data = total_reliance_data[0]
    normed_reliance_data = unpacker.normalize_data(reliance_data)

    # SPLITTING THE DATA and NORMALIZING
    train_data, validation_data, train_target, validation_target = unpacker.split_data(normed_reliance_data, "Close", ratio=0.8)
    normed_total_data = [unpacker.normalize_data(data) for data in [train_data, validation_data, train_target, validation_target] ]

    # CONVERTING TO TENSORS
    normed_tensor_data = [unpacker.to_tensor(data) for data in normed_total_data]














