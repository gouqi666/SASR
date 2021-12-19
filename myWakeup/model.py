import torch
from torch import nn


# class Wake_UP_Model(nn.Module):
#     def __init__(self):

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, target_size=2):
        super(DNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 10000)
        self.hidden3 = nn.Linear(10000, 5000)
        self.hidden4 = nn.Linear(5000, 500)
        self.hidden5 = nn.Linear(500, 50)
        self.hidden6 = nn.Linear(50, 10)
        self.hidden7 = nn.Linear(10, target_size)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.hidden1(x))
        out = self.act(self.hidden2(out))
        out = self.act(self.hidden3(out))
        out = self.act(self.hidden4(out))
        out = self.act(self.hidden5(out))
        out = self.act(self.hidden6(out))
        return self.hidden7(out)
