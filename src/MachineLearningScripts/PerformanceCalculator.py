'''
This class will contian methods to evaulate a model predictive ability by finding different measure of accuracy


'''
import numpy as np
from typing import Tuple

class PerformanceCalculator:
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

    def mse(self) -> float:
        return np.mean((self.y_true - self.y_pred)**2)

    def rmse(self) -> float:
        return np.sqrt(self.mse())

    def mae(self) -> float:
        return np.mean(np.abs(self.y_true - self.y_pred))

    def r2(self) -> float:
        ss_res = np.sum((self.y_true - self.y_pred)**2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true))**2)
        return 1 - ss_res/ss_tot
    
    def totalCorrect(self) -> Tuple[int, float]:
        totalCorrect = 0
        for yTrue, yPred in zip(self.y_true, self.y_pred):
            if yTrue == yPred:
                totalCorrect += 1   
        
        return totalCorrect, totalCorrect / len(self.y_true)
        

