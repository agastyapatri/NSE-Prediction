"""
Python File Containing the Configurations required by the main module
"""

class Configure:
    """
    Configuring the predictive model in one place.
    stock_data: configs to load the stock data
    unpacker: configs to unpack the stock data
    plotter: configs for visualizing the data
    model: configs for the network
    """

    def configurations(self):
        config = {

            "stock_data": {
                "ticker": None,
                "earliest_date": None,
                "latest_date": None
            },

            "unpacker": {
                "path": None,
                "ratio": None,

            },


            "network": {
                # configs for the network, agnostic to the architecture of the network.
                "input_size": None,
                "output_size": None,
                "hidden_sizes": None,
            },

            "model": {
                # configs for the model that will train the network
                "num_epochs": None,
                "learning_rate": None,
                "num_batches": None,
            }

        }
        return config

    def testmethod(self):
        print("Configure is imported")


if __name__ == "__main__":
    test_config = Configure()
    a = test_config.configurations()