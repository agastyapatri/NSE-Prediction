"""
Class to enable the visualization of data. This module also enables the use of the Alpha Vantage Stock
Price API.
"""

import numpy as np
import torch
import torch.nn as nn
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

    # configurations for alphavantage


    def __init__(self, PATH, ticker):
        super().__init__()

        self.PATH = PATH
        self.ticker = ticker




    def alpha_vantage_data(self, outputsize):
        """
        Function to access the Alpha Vantage stock data API, and return a Dataframe
        """

        ts = TimeSeries(key = self.ticker)
        data, meta_data = ts.get_daily(self.ticker, outputsize="full")

        dates = list(data.keys())
        values = list(data.values())

        dataframe = pd.DataFrame(columns = ["Open", "High", "Low", "Close", "Volume"], index=dates, dtype=np.float64)

        for i in range(len(values)):
            vals = list(values[i].values())
            dataframe.iloc[i,:] = vals
        dataframe = dataframe.astype(np.float64)


        return dataframe


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



    def to_tensor(self, dataframe, normalize=False):
        """
        Function to convert the pandas DataFrame Object to pytoch Tensor object
        :return: Tensors and arrays
        """
        tensor = torch.tensor(dataframe.values, dtype=torch.float64)
        if normalize==True:
            tensor = functional.normalize(tensor, p=2.0, dim=1)

        return tensor



    def split_data(self, stock_data, target, ratio):
        """
        Function to yield train and validation data from the input. Also returns the target variable
        :param target_index: column index of the feature desired to be dropped.
        :param ratio: the ratio intended in the train-test-split
        :return: train_data, validation_data, train_target, validation_target
        """
        stock_tensor = self.to_tensor(stock_data, normalize=True)


        feature_data = stock_data.drop([target], axis = 1)
        target_data = stock_data[target]
        train_indices = int(ratio * len(stock_data))

        # converting to tensors
        feature_tensor = self.to_tensor(feature_data, normalize=True)
        target_tensor = self.to_tensor(feature_data, normalize=True)

        train_data_tensor = feature_tensor[:train_indices, :]
        validation_data_tensor = target_tensor[train_indices:, :]

        train_target_tensor = target_tensor[:train_indices]
        validation_target_tensor = target_tensor[train_indices:]



        return train_data_tensor, validation_data_tensor, train_target_tensor, validation_target_tensor




    def normalize_data(self, data):
        """
        Function to normalize the data fed into it
        :param data: torch tensors
        :return:
        """
        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)
            normed_data = (data  - data.mean()) / data.std()

        elif isinstance(data, torch.Tensor):
            normed_data = torch.mean(data)[0]

        return data




    def testmethod(self):
        print("unpack_data is imported")




if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------------------------------
    Testing the methods of the class, 
    -----------------------------------------------------------------------------------------------------------------"""

    unpacker = Unpacking(PATH=None, ticker="IBM")
    # LOADING THE DATA

    total_ibm_data = unpacker.alpha_vantage_data(outputsize="full")

    # train_data, validation_data, train_target, validation_target
    train_test_split = unpacker.split_data(stock_data=total_ibm_data, target="Close", ratio=0.8)







