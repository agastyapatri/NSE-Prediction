#  _Stock Market Prediction_ 




_This project is inspired by the papers:_
1. M, H., E.A., G., Menon, V.K. and K.P., S. (2018). NSE Stock Market Prediction Using Deep-Learning Models. Procedia Computer Science, 132, pp.1351–1362. doi:10.1016/j.procs.2018.05.050.


2. Lu, W., Li, J., Li, Y., Sun, A. and Wang, J. (2020). A CNN-LSTM-Based Model to Forecast Stock Prices. _ Complexity_, 
2020, pp.1–10. doi:10.1155/2020/6622927.

_This project will stay under development for the considerable future._
## 1. Introduction 
* The paper uses 4 types of deep learning architectures to predict
  the stock price of a company based on historical prices.
    1. Multilayer Perceptron (**MLP**)
    2. Long Short-Term Memory **(LSTM)**
    3. Convolutional Neural Network **(CNN)**


* This project will focus on the following stocks:
  1. Reliance Industries **(RELIANCE)**
  2. Tata steel (**TATASTEEL**)
  3. Godrej **(GODREJIND)**
  4. Bajaj finance **(BAJFINANCE)**
  5. Tata motors **(TATAMOTORS)**
  6. Larsen Toubro **(LT)**


* Each dataset consists of 5 columns:
  1. **Open:** _Stock price at the beginning of the trading day._
  2. **High:** _Highest price reached by the stock._ 
  3. **Low:** _Lowest Price reached by the stock._
  4. **Close:** _Price of the stock at the end of the trading day._
  5. **Adjusted Close:** _Close price adjusted for stock splits and other caveats._

## 2. Methodology of prediction 
From the available stock data, the day-wise closing price of the stock, as investment
decisions are made based on the performance of the stock at the end of the trading period.

A suite of different models are defined: 
  1. Multilayer Perceptrons (**MLP**)
  2. Convolutional Neural Networks (**CNN**)
  3. Long Short Term Memory Networks (**LSTM**)

Following are a few caveats:

1. The training data is normalied to be between -1 and 1.
2. The output of the model is subjected to a de-normalization process to acquire original predicted values. 
3. The paper uses the following parameters:
   * Window Size = 20
   * Number of Epochs = 100
4. To check if the model is able to generalize common stock market
   dynamics, the model is also tested on the **NYSE**

* The code was written as generally as possible, with a trial application to RELIANCE.



## 3. Results
This section details the performance of each model architecture. Based on the metrics, chosen 
it is expected that the **LSTM** will perform optimally.

## 3. Notes
* There was no need to flatten the tensors, because each 
  feature vector is a 1-D array. 
