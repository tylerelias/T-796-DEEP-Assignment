import torch
import numpy as np
import model
from sklearn.model_selection import train_test_split


class Predictor:
    def __init__(self, name, model, args=None):
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

    def normalize_single(self, info_company, info_quarter, info_daily, current_stock_price):
        COMPANY_NAME = 0
        YEAR = 1
        DAY = 2
        QUARTER = 3
        EXPERT_1 = 4
        # EXPERT_2 index not used, skip
        SENTIMENT = 6
        M1 = 7
        M2 = 8
        M3 = 9
        M4 = 10
        STOCK_PRICE = 11
        SEGMENT = 12
        TREND = 13

        combined_data = []

        # Store the Max value for each data
        max_stock_price = 178.0
        max_day = 365
        max_sentiment_analysis = 10
        max_m1 = 10.0
        max_m2 = 9994.0

        for i in range(len(current_stock_price)):

            ms_index = tuple()  # Market segment index, instead of string
            trend = tuple()  # Trend value, -1 to 1

            if info_company[info_daily[i][0]][1] == 'IT':
                ms_index = ms_index + (1,)
                # Look up the trend value for the given segment
                for item in info_quarter:
                    list_item = list(item)
                    if (info_daily[i][YEAR]) == list_item[1] and \
                            info_daily[i][QUARTER] == list_item[2] and \
                            list_item[0] == 'IT':
                        trend = trend + (item[3],)
                        break

            elif info_company[info_daily[i][0]][1] == 'BIO':
                ms_index = ms_index + (0,)
                # Look up the trend value for the given segment
                for item in info_quarter:
                    list_item = list(item)
                    if (info_daily[i][YEAR]) == list_item[1] and \
                            info_daily[i][QUARTER] == list_item[2] and \
                            list_item[0] == 'BIO':
                        trend = trend + (item[3],)
                        break

            # Place all the data into a single list
            # juck
            super_tuple = list(info_daily[i])
            super_tuple.append(current_stock_price[i])
            super_tuple = tuple(super_tuple)
            combined_data.append(
                super_tuple +
                ms_index +
                trend
            )

        normalized = []

        for s in combined_data:
            replace = (
                s[COMPANY_NAME],  # company name
                s[YEAR] - 2017,  # year
                s[DAY] / max_day,  # day
                s[QUARTER],  # quarter
                s[STOCK_PRICE] / max_stock_price,  # stock price
                s[EXPERT_1],  # expert 1
                s[SENTIMENT] / max_sentiment_analysis,  # sentiment analysis
                s[M1] / max_m1,  # m1
                s[M2] / max_m2,  # m2
                s[M3],  # m3
                s[M4],  # m4
            )
            normalized.append(replace)

        return normalized

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

        normalized = self.normalize_single(info_company, info_quarter, info_daily, current_stock_price)

        WIDTH = 56  # The width of the layers
        IN_FEATURES = np.array(normalized).shape[1]  # The amount of data coming in [x, y, z, a]
        OUT_FEATURES = 2  # Number of possible answers [0, 1]
        DROPOUT_VALUE = 0.375

        cross_net = model.MyANN(DROPOUT_VALUE, IN_FEATURES, WIDTH, OUT_FEATURES)
        cross_net.load_state_dict(torch.load(self.name_))

        answers = []

        for i in range(len(normalized)):
            X = torch.Tensor(normalized[i])
            cross_net.eval()
            output = cross_net(X)
            prediction = torch.argmax(output)
            print(prediction)
            if prediction == 0:
                answers.append(False)
            else:
                answers.append(True)

        return answers[0], answers[1], answers[2]
