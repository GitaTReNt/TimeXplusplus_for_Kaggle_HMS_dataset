import torch

class DatasetwInds(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X
        self.times = times
        self.y = y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y, torch.tensor(idx).long().to(x.device)


class DatasetwInds1(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X
        self.times = times
        self.y = y

    def __len__(self):
        return self.X.shape[0]  # 返回样本数量

    def __getitem__(self, idx):
        x = self.X[idx, :, :]  # 索引样本
        T = self.times[idx, :]  # 索引样本对应的时间轴
        y = self.y[idx]  # 索引样本对应的标签
        return x, T, y, torch.tensor(idx).long().to(x.device)
