import numpy as np

class GLM:

    def __init__(self):

        self.parameters: np.ndarray = np.ndarray([])

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.parameters.size == 0:
            print("Model has not been trained yet")
            return 

        return self.parameters @ X
        
    '''
    Gaussian  regression 
    (note) use my actual linear regression class for more choices of learning styles if for some reason you didn't want to use OLS
    '''
    def GaussianGLM(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        res = np.linalg.inv(X1.T @ X1) @ X1.T @ Y
        
        self.parameters = res

        return res
    
    '''
    Binomial (logistic) regression 
    (note) use my actual logistic regression class for more choices
    '''
    def BinomialRegression(self, X: np.ndarray,Y: np.ndarray, LearningRate: float, Epochs: int, BatchSize: int) -> np.ndarray:

        #needs sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        # 1 / e^-eta
        # theta = theta - learning rate (Y - h(x)) x^i

        if self.parameters.size[1] == 0:
            self.parameters = np.random.randn(X1.size[1])

        for _ in range(Epochs):

            for x,y in zip(X1,Y):
                graident = (y - sigmoid(x @ self.parameters)) * x
                self.parameters = self.parameters + LearningRate * graident

        return self.parameters
    
    '''
    Binomial (logistic) regression 
    (note) use my actual logistic regression class for more choices
    '''
