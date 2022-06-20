"""
Class to enable the visualization of data
"""

import numpy as np
import torch
from torch import nn
import pandas as pd
import os


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


    def split_data(self, stock_data, ratio):
        """
        Function to yield train and validation data from the input
        :input: ratio: the ratio of train data to total data
        :return: stock_data (DataFrame), train_data (DataFrame), validation_data (DataFrame)
        """
        train_indices = int(ratio * len(stock_data))

        train_data = stock_data.iloc[:train_indices, 1:]
        validation_data = stock_data.iloc[train_indices:, 1:]

        return stock_data, train_data, validation_data


    def to_tensor(self, stock_data, ratio):
        """
        Function to convert the pandas DataFrame Object to pytoch Tensor obeject. Takes in a DataFrame
        :return: Tensors and arrays
        """
        stock_data, train_data, validation_data = self.split_data(stock_data, ratio)

        train_data = torch.tensor(np.array(train_data.values), dtype=torch.float32)
        validation_data = torch.tensor(np.array(validation_data.values), dtype=torch.float32)
        stock_tensor = torch.tensor(np.array(stock_data), dtype=torch.float32)

        return stock_tensor, train_data, validation_data

    @staticmethod
    def testmethod():
        print("Import Successful")

if __name__ == "__main__":

    """
    Testing the methods of the class, and the relations between the tensors
    """

    unpack_test = Unpacking(PATH = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/", ticker = "RELIANCE")
    test_data = unpack_test.load_data()[0]
    test_tensors = unpack_test.to_tensor(test_data, 0.8)







