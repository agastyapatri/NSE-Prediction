"""
Class to train any model given to it.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional



class Train(nn.Module):
    device = torch.device("cpu")
    dtype  = torch.float

    def __init__(self, num_epochs, learning_rate, optimizer, criterion):
        super(Train, self).__init__()

        """
        Defining the parameters which are common to all functions. Defining the criterion and the 
        optimizer in such a way will allow for easier hyperparameter optimization 
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # defining the loss criterion
        if criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "CEL":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == "NLL":
            self.criterion = nn.NLLLoss()
        else:
            raise Exception("Not a valid Criterion. Choose between MSE, CEL and NLL")


        # defining the optimizer
        if optimizer == "Adam":
            self.optimizer = lambda network : torch.optim.Adam(params=network.parameters(),
                                                               lr=self.learning_rate)
        elif optimizer == "SGD":
            self.optimizer = lambda network : torch.optim.SGD(params=network.parameters(),
                                                              lr = self.learning_rate)
        else:
            raise Exception("Not a valid optimizer. Choose between Adam and SGD")




    # Function to train any model that is fed to it
    def trainmodel(self, network, train_data, train_labels, save = None):

        """
        :param network: The network used to predict
        :param train_data: train data
        :param train_labels: labels for the train data

        :return: trained model
        """


        train_accuracy_array = []
        y_true = train_labels
        X = train_data
        optim = self.optimizer(network)
        correct_count = 0
        incorrect_count = 0


        # Iterating over all the epochs
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            y_pred = network(X)

            # setting the gradients at the nodes of the computational graph = 0.
            optim.zero_grad()

            # Calculating loss
            loss = self.criterion(y_pred, y_true)

            # backward pass: computing gradients
            loss.backward(retain_graph=True)

            with torch.no_grad():

                # manually updating the weights: gradient descent through the computational graph
                optim.step()

            running_loss += loss.item()

            # Calculating the training accuracy
            bool_array = np.array(y_pred == y_true)
            num_correct, num_incorrect = np.count_nonzero(bool_array), len(train_data) - np.count_nonzero(bool_array)
            accuracy = num_correct/(num_correct + num_incorrect)
            train_accuracy_array.append(accuracy)

            # printing the loss at every 100th epoch
            if epoch % 100 == 99:
                print(f"Epoch {epoch}: Train Loss = {running_loss/len(train_data)}; Train Accuracy = {accuracy}")




    @staticmethod
    def testmethod():
        print("trainer is imported")






if __name__ == "__main__":
    """-----------------------------------------------------------------------------------------------------------------
    Testing the Train class
    ~ traindata and testdata defined below are one sample of a dataset that has 1000 features. (1 x 1000)
    -----------------------------------------------------------------------------------------------------------------"""
    testnetwork = nn.Sequential(
        # Input Layer -> Hidden Layer
        nn.Linear(1000, 500, bias=True),
        nn.ReLU(),

        # Hidden Layer -> Output Layer
        nn.Linear(500, 1, bias=True),
    )


    # train_data: 100 samples, 1000 features/sample. train_labels: 100 labels.
    traindata = torch.randn(100, 1000)
    trainlabels = torch.zeros(100, 1)


    # randomly sampling data from the normal distribution of (mu, sigma) = (1, 0.1)
    testdata = torch.tensor(np.random.normal(1, 0.1, 10000)).float()
    testlabels = torch.zeros(1)


    # defining the trainer
    trainer = Train(num_epochs=1000, learning_rate=1e-6, optimizer="SGD", criterion="MSE")
    a = trainer.trainmodel(network=testnetwork, train_data=traindata, train_labels=trainlabels, save = None)

