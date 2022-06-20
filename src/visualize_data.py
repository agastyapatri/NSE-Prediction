"""
Python Class to visualize the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpl
import matplotlib.dates as mdts
import os



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

    def __init__(self, stock_data, ticker, features, start_date, end_date):
        self.stock_data = stock_data
        self.ticker = ticker
        self.features = features
        self.start_date = start_date
        self.end_date = end_date


    def visualize(self):
        """
        Function to cast the data into a usable format
        :return: Required Plots
        """
        self.stock_data.index = pd.DatetimeIndex(self.stock_data.Date)

        start = self.stock_data.index.searchsorted(self.start_date)
        end = self.stock_data.index.searchsorted(self.end_date)

        plt.title(f"Performance vs Time for the stock {self.ticker} from {self.start_date} to {self.end_date}")
        for i in range(len(self.features)):
            feature = self.features[i]
            plt.plot(self.stock_data.index[start:end], self.stock_data[feature].iloc[start:end],
                     label = f"{feature} vs Time")
        plt.legend()
        plt.show()




    def OHLC(self):
        """
        Function to plot the candlestick / OHLC graph
        """

        self.stock_data.index = pd.DatetimeIndex(self.stock_data.Date)

        start = self.stock_data.index.searchsorted(self.start_date)
        end = self.stock_data.index.searchsorted(self.end_date)

        mpl.plot(self.stock_data.iloc[start:end], type = "candle", mav = (3,6,9),
                 title = f"OHLC Price of {self.ticker} from {self.start_date} to {self.end_date}",
                 style = "starsandstripes")

        plt.xlabel("Time (Years)")
        plt.ylabel("Price")
        plt.show()

        


    def comparisons(self):
        """
        Function to see the true growth of the feature, vs the predicted growth of the feature.
        :return: Plots
        """
        pass
    

    @staticmethod
    def testmethod():
        print("import successful")






if __name__ == "__main__":
    """
    Function to test the methods in the class above
    """

    PATH = "/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/data/"
    ticker = "RELIANCE"
    file = ticker + ".NS.csv"

    test_data = pd.read_csv(os.path.join(PATH, file))


    test = Visualizer(stock_data=test_data, ticker=ticker, features=["Adj Close", "Close"],
                      start_date="2021-01-01", end_date="2022-02-01")


