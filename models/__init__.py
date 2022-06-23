"""
Initialization of the "models" package
"""


import sys
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/models/__init__.py")
sys.path.append("/NSE_Prediction/models/FFN.py")
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/models/CNN.py")
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/models/RNN.py")


from .FFN import *
from .RNN import *
from .CNN import *