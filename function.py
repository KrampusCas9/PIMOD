from scipy import stats
import numpy as np
import torch

class LabelProcess:
    def __init__(self, train_labels):
        self.label = torch.tensor(train_labels, dtype=torch.float32)
        self.mean = torch.mean(self.label, dim=0)
        self.std = torch.std(self.label, dim=0)
        self.min = -80.0
        self.max = 80.0
    
    def standardize(self, labels):
        device = labels.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return (labels - mean) / std
    
    def inverse_standardize(self, labels):
        if isinstance(labels, torch.Tensor):
            device = labels.device
            mean = self.mean.to(device)
            std = self.std.to(device)
            return labels * std + mean
        elif isinstance(labels, np.ndarray):
            std = self.std.cpu().numpy()
            mean = self.mean.cpu().numpy()
            return labels * std + mean
        else:
            raise TypeError("Input should be a torch.Tensor or numpy.ndarray")

    def normalize(self, labels):
        return (labels - self.min) / (self.max - self.min)
    
    def inverse_normalize(self, labels):
        return labels * (self.max - self.min) + self.min

def caculate(y_pred, y):
    y = y.flatten()
    y_pred = y_pred.flatten()

    if y.shape != y_pred.shape:
        raise ValueError("y and y_pred must have the same shape")

    corr = np.corrcoef(y, y_pred)[0, 1]
    if np.isnan(corr):
        corr = np.corrcoef(y, y_pred)[1, 0]

    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / y.mean()
    mae = np.mean(np.abs(y - y_pred))

    return mse, mae, corr, nrmse

def transform_zscore(array):
    mean = np.mean(array)
    std_dev = np.std(array)
    normalized_array = stats.zscore(array)
    return normalized_array, mean, std_dev

def inverse_transform_zscore(array, mean, std_dev):
    restored_array = normalized_array * std_dev + mean
    return restored_array
