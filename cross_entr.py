# Imports

import torch.nn as nn
import predictor
import simulator

predictor = predictor.Predictor("cross_network_0.915", None)
simulator.simulate(2009, 0, predictor)


def read_stock_prices(stocks):
    with open('data/stock_prices.txt') as f:
        for line in f:
            fields = line.split()

            year = 0
            # Making the years into 0-2 format, put in dict later?
            if fields[1] == "2017":
                year = 0
            if fields[1] == "2018":
                year = 1
            if fields[1] == "2019":
                year = 2

            record = (int(fields[0]),
                      int(year),
                      int(fields[2]),
                      int(fields[3]),
                      float(fields[4]))
            stocks.append(record)


def read_info(info):
    with open('data/info.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),
                      int(fields[4]),
                      int(fields[5]),
                      int(fields[6]),
                      float(fields[7]),
                      float(fields[8]),
                      float(fields[9]),
                      int(fields[10]),)
            info.append(record)


def read_segments(segments):
    with open('data/market_segments.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      str(fields[1]),)
            segments.append(record)


def read_market_analysis(analysis):
    with open('data/market_analysis.txt') as f:
        for line in f:
            fields = line.split()
            record = (str(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),)
            analysis.append(record)


# CONSTANTS FOR COLUMN NAMES
COMPANY_NAME = 0
YEAR = 1
DAY = 2
QUARTER = 3
STOCK_PRICE = 4
EXPERT_1 = 5
# EXPERT_2 index not used, skip
SENTIMENT = 7
M1 = 8
M2 = 9
M3 = 10
M4 = 11
SEGMENT = 12
TREND = 13

current_stock_price = []
info_daily = []
info_company = []
info_quarter = []

combined_data = []

# Store the Max value for each data
max_stock_price = 0
max_day = 0
max_sentiment_analysis = 0
max_m1 = 0
max_m2 = 0

# The target predictions for the ANN
target_label = []

# Read from the files
read_stock_prices(current_stock_price)
read_info(info_daily)
read_segments(info_company)
read_market_analysis(info_quarter)

# This is used to compare prev. stock price to current one
previous = tuple()

for i in range(len(current_stock_price)):

    ms_index = tuple()  # Market segment index, instead of string
    trend = tuple()  # Trend value, -1 to 1

    if info_company[current_stock_price[i][0]][1] == 'IT':
        ms_index = ms_index + (1,)
        # Look up the trend value for the given segment
        for item in info_quarter:
            list_item = list(item)
            if (current_stock_price[i][YEAR] + 2017) == list_item[1] and \
                    current_stock_price[i][QUARTER] == list_item[2] and \
                    list_item[0] == 'IT':
                trend = trend + (item[3],)
                break


    elif info_company[current_stock_price[i][0]][1] == 'BIO':
        ms_index = ms_index + (0,)
        # Look up the trend value for the given segment
        for item in info_quarter:
            list_item = list(item)
            if (current_stock_price[i][YEAR] + 2017) == list_item[1] and \
                    current_stock_price[i][QUARTER] == list_item[2] and \
                    list_item[0] == 'BIO':
                trend = trend + (item[3],)
                break

    # Place all the data into a single list
    combined_data.append(
        current_stock_price[i] +
        info_daily[i][STOCK_PRICE::] +
        ms_index +
        trend
    )

    # Create a lable to use for the model
    if previous and previous[STOCK_PRICE] < current_stock_price[i][STOCK_PRICE]:
        target_label.append(1)
    else:
        target_label.append(0)

    # See if there is a new max stock prize
    if combined_data[i][STOCK_PRICE] > max_stock_price:
        max_stock_price = combined_data[i][STOCK_PRICE]

    if combined_data[i][DAY] > max_day:
        max_day = combined_data[i][DAY]

    if combined_data[i][SENTIMENT] > max_sentiment_analysis:
        max_sentiment_analysis = combined_data[i][SENTIMENT]

    if combined_data[i][M1] > max_m1:
        max_m1 = combined_data[i][M1]

    if combined_data[i][M2] > max_m2:
        max_m2 = combined_data[i][M2]
    # To compare the current stock with previous stock
    previous = combined_data[i]

# Normalize the data

normalized = []

for s in combined_data:
    replace = (
        s[COMPANY_NAME],  # company name
        s[YEAR],  # year
        s[DAY] / max_day,  # day
        s[QUARTER],  # quarter
        s[STOCK_PRICE] / max_stock_price,  # stock price
        s[EXPERT_1],  # expert 1
        s[SENTIMENT] / max_sentiment_analysis,  # sentiment analysis
        s[M1] / max_m1,  # m1
        s[M2] / max_m2,  # m2
        s[M3],  # m3
        s[M4],  # m4
        s[SEGMENT],  # market_segment, just seems to decrease accuracy
        s[TREND],  # trend

    )
    normalized.append(replace)

# Read test and training data from files into a dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 10

X, y = torch.Tensor(normalized), torch.Tensor(target_label)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_set = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_set = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

WIDTH = 64  # The width of the layers
IN_FEATURES = X.shape[1]  # The amount of data coming in [x, y, z, a]
OUT_FEATURES = 2  # Number of possible answers [0, 1]
EPOCHS_VAL = 90
LEARNING_RATE = 0.00515
DROPOUT_VALUE = 0.375


class MyANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_VALUE)

        self.fc1 = nn.Linear(IN_FEATURES, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc3 = nn.Linear(WIDTH, WIDTH)
        self.fc4 = nn.Linear(WIDTH, WIDTH)
        self.fc5 = nn.Linear(WIDTH, OUT_FEATURES)
        return

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x)))
        x = torch.relu(self.dropout(self.fc4(x)))
        x = self.fc5(x)
        return nn.functional.log_softmax(x, dim=1)


net = MyANN()

for _ in range(100):

    net = MyANN()
    optimization = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.train()
    for epochs in range(EPOCHS_VAL):
        for data in train_set:
            _X, _y = data
            net.zero_grad()
            output = net(_X)  # Forward pass
            _y = torch.tensor(_y, dtype=torch.long)
            loss_c = nn.CrossEntropyLoss()
            loss = loss_c(output, _y)  # Computing
            loss.backward()  # Back-propigation
            optimization.step()

    # Evaluate
    total = 0
    correct = 0
    net.eval()
    for data in test_set:
        _X, _y = data
        output = net(_X)  # Forward pass
        for idx, val in enumerate(output):
            if torch.argmax(val) == _y[idx]:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))

    if round(correct / total, 3) >= 0.9:  # Save all networks that give > 90% acc, muhaha
        filename = "network_" + str(round(correct / total, 3)) + ".pt"
        print(filename)
        torch.save(net.state_dict(), filename)
