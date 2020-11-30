import torch
from torch import nn

class MyANN(nn.Module):
    def __init__(self, DROPOUT_VALUE, IN_FEATURES, WIDTH, OUT_FEATURES):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_VALUE)

        self.fc1 = nn.Linear(IN_FEATURES, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc3 = nn.Linear(WIDTH, WIDTH)
        self.fc4 = nn.Linear(WIDTH, WIDTH)
        self.fc5 = nn.Linear(WIDTH, OUT_FEATURES)
        return

    def forward(self, x):
        # Aware that this call to relu is not a standard way
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x)))
        x = torch.relu(self.dropout(self.fc4(x)))
        x = self.fc5(x)
        return nn.functional.log_softmax(x, dim=1)
