import numpy as np


class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None


    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_true - y_pred)**2)
        return loss



class BCELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
    
        self.y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        self.y_true = y_true

        loss = -np.mean(self.y_true*np.log(self.y_pred) + (1 - self.y_true)*np.log(1 - self.y_pred))
        return loss


class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
    
        
        
        self.y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        self.y_true = y_true

        loss = -np.mean(self.y_true*np.log(self.y_pred) + (1 - self.y_true)*np.log(1 - self.y_pred))
        return loss
