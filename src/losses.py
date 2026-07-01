import numpy as np


class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None


    def forward(self, y_pred, y_true):
        loss = np.mean((y_true - y_pred)**2)
        return loss

    def backward(self):
        pass


