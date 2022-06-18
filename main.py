"""
Main Class of the project
"""
import torch
from src.visualize_data import Visualizer

test_data = torch.randn(5,6)

visualizer = Visualizer(stock_data=None, ticker=None, feature=None, start_date=None, end_date=None)
visualizer.testmethod()

