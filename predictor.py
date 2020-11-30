class Predictor:
    def __init__(self, name, model, args = None):
        """
        Constructor
        :param name:  A name given to your predictor
        :param model: An instance of your ANN model class.
        :param parameters: An optional dictionary with parameters passed down to constructor.
        """
        self.name_ = name
        self.model_ = model
        #
        # You can add new member variables if you like.
        #
        return

    def get_name(self):
        """
        Return the name given to your predictor.
        :return: name
        """
        return self.name_

    def get_model(self):
        """
         Return a reference to you model.
         :return: model
         """
        return self.model_

    def predict(self, info_company, info_quarter, info_daily, current_stock_price):
        """
        Predict, based on the most recent information, the development of the stock-prices for companies 0-2.
        :param info_company: A list of information about each company
                             (market_segment.txt records)
        :param info_quarter: A list of tuples, with the latest quarterly information for each market sector.
                             (market_analysis.txt records)
        :param info_daily: A list of tuples, with the latest daily information about each company (0-2).
                             (info.txt records)
        :param current_stock_price: A list of floats, with the with the current stock prices for companies 0-2.

        :return: A Python 3-tuple with your predictions: go-up (True), not (False) [company0, company1, company2]
        """
        #
        # Add your code here ...
        #
        return False, False, False
