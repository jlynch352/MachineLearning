''' 
Linear Regression From Scrath

File Contiains
1. Prediction
2. Ordinary Least Squares Estimation
3. Stochastic Gradient Descent
4. Batch Gradient Descent 
5. Stochastic Batch Gradient Descent
6. Locally Weighted Regression
'''
import numpy as np


class LinearRegression:
    def __init__(self):
        self.parameters: np.ndarray = np.array([])


    def Prediction(self, X) -> np.ndarray:
        if self.parameters.size == 0:
            print("You haven't trained the model yet")
            return np.array([])
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        return X1 @ self.parameters

    def OrdinaryLeastSquaresMethod(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        res = np.linalg.inv(X1.T @ X1) @ X1.T @ Y
        
        self.parameters = res

        return res
    
    def StochasticGradientDescent(self, X:np.ndarray,Y:np.ndarray, LearningRate: float, Epochs: int) -> np.ndarray:
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])

        for i in range(Epochs):
            for x,y in zip(X1,Y):
                prediction = x @ self.parameters
                graident = (prediction - y) * x
                self.parameters = self.parameters - (LearningRate * graident)
        
        return self.parameters
    
    def BatchGradientDescent(self, X: np.ndarray, Y: np.ndarray, LearningRate: float, Epochs: int) -> np.ndarray:
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])
    
        for _ in range(Epochs):
            gradient = np.zeros_like(self.parameters)  

            for x, y in zip(X1, Y):
                prediction = x @ self.parameters
                gradient += (prediction - y) * x

            gradient /= X1.shape[0]
            self.parameters -= LearningRate * gradient

        return self.parameters

    def StochasticBatchGradientDescent(self, X: np.ndarray, Y: np.ndarray, LearningRate: float, Epochs: int, batchSize: int) -> np.ndarray:
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])
        
        n = X1.shape[0]

        for _ in range(Epochs):
            index = np.random.randint(0, n - batchSize + 1)

            X_batch = X1[index:index + batchSize]
            Y_batch = Y[index:index + batchSize]
            
            gradient = np.zeros_like(self.parameters)

            for x, y in zip(X_batch,Y_batch):
                prediction = x @ self.parameters
                gradient += (prediction - y) * x

            gradient /= batchSize
            self.parameters -= LearningRate * gradient
        
        return self.parameters
            
    def LocallyWeightedRegression(self,X, Y, x1, tau) -> np.ndarray:


        #matrix of difference of between the data points and the target point
        difference = X - x1

        #sqaured distance from data point
        squaredDistance = np.sum(difference**2,axis = 1) 

        weights = np.exp(-squaredDistance / (2 * tau ** 2))

        W_X = weights[:, None] * X
        Y_X = weights * Y      

        self.parameters = np.linalg.inv(X.T @ W_X) @ (X.T @ Y_X)

        return self.parameters

        



        
        
        
        
            

