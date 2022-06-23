#  _Stock Market Prediction_ 

_This project is inspired by the papers:_
1. Hiransha M et al. "NSE Stock Market Prediction Using Deep-Learning Models", ICCIDS 2018.
2. W Lu et al. "A CNN-LSTM-Based Model to Forecast Stock Prices", Wiley 2020.

_This project will stay under development for the considerable future._
## 1. Introduction 
* The paper uses 4 types of deep learning architectures to predict
  the stock price of a company based on historical prices.
    1. Multilayer Perceptron (**MLP**)
    2. Recurrent Neural Network **(RNN)**
    3. Long Short-Term Memory **(LSTM)**
    4. Convolutional Neural Network **(CNN)**


* This project will focus on the following stocks:
  1. Reliance Industries **(RELIANCE)**
  2. Tata steel (**TATASTEEL**)
  3. Godrej **(GODREJIND)**
  4. Bajaj finance **(BAJFINANCE)**
  5. Tata motors **(TATAMOTORS)**
  6. Larsen Toubro **(LT)**

* Each dataset consists of 5 columns:
  1. Open
  2. High 
  3. Low
  4. Close
  5. Adjusted Close

## 2. Methodology of prediction 
From the available stock data, the day-wise closing price of the stock, as investment
decisions are made based on the performance of the stock at the end of the trading period.

**The data was obtained from the NASDAQ data link**

1. The training data is normalied to be between -1 and 1.
2. The output of the model is subjected to a de-normalization process to acquire original predicted values. 
3. The paper uses the following parameters:
   * Window Size (**?????**) = 200
   * Number of Epochs = 1000
4. To check if the model is able to generalize common stock market
   dynamics, the model is also tested on the **NYSE**

* The code was written as generally as possible, with a trial application to RELIANCE.


## 3. Notes
* There was no need to flatten the tensors, because each 
  feature vector is a 1-D array. 
