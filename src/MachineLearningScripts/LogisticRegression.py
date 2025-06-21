import numpy as np


class LogisticRegression:
    def __init__(self, threshold: float = 0.5):
         self.parameters: np.ndarray = np.array([])
         self.threshold = threshold
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -500,  500)
        return 1 / (1 + np.exp(-z))

    def sigmoidDerivative(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid(x)
        return s*(1-s)
    
    def PredictionProbability(self, X) -> np.ndarray:
        if self.parameters.size == 0:
            print("You haven't trained the model yet")
            return np.array([])
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        return self.sigmoid(X1 @ self.parameters)
    
    def PredictionClass(self,X) -> np.ndarray:

        probs = self.PredictionProbability(X)
        return (probs >= self.threshold).astype(int)
    
    def StochasticGraidentDescent(self, X, Y, LearningRate: float, Epochs: int) -> np.ndarray:

        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])

        for _ in range(Epochs):
            
            for x,y in zip(X1,Y):
                graident = (y - self.sigmoid(x @ self.parameters)) * x
                self.parameters = self.parameters + LearningRate * graident

        return self.parameters

    def NewtonsMethod(self, X: np.ndarray, Y: np.ndarray, Delta: float, ridge: float = 1e-4, MaxIterations = 10) -> np.ndarray:

        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])

        for _ in range(MaxIterations):

        # 1) scores and probabilities
            z = X1 @ self.parameters
            probs = self.sigmoid(z)

        # 2) gradient of negative log-likelihood
            gradient = X1.T @ (probs - Y)

        # 3) diagonal weight matrix
            W = np.diag(probs * (1 - probs))  

        # 4) Hessian and Newton step
            H = X1.T @ W @ X1
            #add a ridge to ensure invertiblity
            H += ridge * np.eye(H.shape[0])

            step = np.linalg.solve(H, gradient)
        
            self.parameters -= step

            #condition for accuracy
            if np.linalg.norm(step) < Delta:
                return self.parameters

        return self.parameters
            


           