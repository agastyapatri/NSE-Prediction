"""
Class to train any model given to it
"""


import numpy as np
import torch
import torch.nn as nn

class Train(nn.Module):
    def __init__(self, num_epochs, learning_rate, optimizer):
        super().__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def trainmodel(self, network):
        """
        Function to train any model that is fed to it
        """
        running_loss = 0.0
        train_accuracy_array = []

        # Iterating over all the epochs
        for epoch in range(len(self.num_epochs)):

            # printing the loss at every 50th epoch
            if epoch % 50 == 49:
                print(f"Training Loss at epoch: {epoch} is {running_loss}")

        pass

    def testmethod(self):
        print("trainer is imported")

if __name__ == "__main__":
    """-----------------------------------------------------------------------------------------------------------------
    Testing the Train class
    -----------------------------------------------------------------------------------------------------------------"""
    trainer = Train(num_epochs=None, learning_rate=None, optimizer=None)

    trainer.testmethod()