"""
Python Class to visualize the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def testmethod(stock_data):
    print(stock_data)
    print("import successful from /src")


class Visualizer:
    """
    path = /path/to/data/file
    stock_data: A csv file.
    ticker: The ticker name of the asset.
    feature: The desired feature to be plotted
    start_date: date FROM which plotting is desired
    end_date: date TO which plotting is desired
    """

    def __init__(self, stock_data, ticker, feature, start_date, end_date):
        self.stock_data = stock_data
        self.ticker = ticker
        self.feature = feature
        self.start_date = start_date
        # self.end_date = end_date

    def visualize(self):
        """
        Function to cast the data into a usable format
        :return: Required Plots
        """
        if (self.stock_data == None):
            print("Module is Imported, this is a test")

        else:
            start = self.stock_data.index.searchsorted(self.start_date)
            end = self.stock_data.index.searchsorted(self.end_date)

            plt.title(f"{self.feature} data for the stock: {self.ticker}")
            self.stock_data[self.feature].iloc[start:end].plot(c="b")
            plt.xlabel("Time")
            plt.ylabel(f"{self.feature}")
            plt.show()

    @staticmethod
    def testmethod():
        print("import successful")





if __name__ == "__main__":
    test = Visualizer(stock_data=None, ticker=None, feature=None, start_date=None, end_date=None)

