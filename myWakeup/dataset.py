import torch
from torch.utils.data import Dataset,DataLoader
class WakeUpDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        return [self.x[item],self.y[item]]
    def __len__(self):
        return len(self.x)