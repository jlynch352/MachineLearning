import numpy as np


class LogisticRegression:
    def __init__(self, threshold: float = 0.5):
         self.parameters: np.ndarray = np.array([])
         self.threshold = threshold
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -500,  500)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidDerivative(self, x: np.ndarray) -> np.ndarray:
        
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def PredictionProbability(self, X) -> np.ndarray:
        if self.parameters.size == 0:
            print("You haven't trained the model yet")
            return np.array([])

        return self.sigmoid(X @ self.parameters)
    
    def PredictionClass(self,X) -> np.ndarray:

        probabilities = self.PredictionProbability(X)

        return (probabilities >= self.threshold).astype(int)
    
    def StochasticGraidentDescent(self, X, Y, LearningRate: float, Epochs: int) -> np.ndarray:

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X.shape[1])

        for _ in range(Epochs):
            
            for x,y in zip(X,Y):
                graident = (y - self.sigmoid(x @ self.parameters)) * x
                self.parameters = self.parameters + LearningRate * graident

        return self.parameters

    def NewtonsMethod(self, X: np.ndarray, Y: np.ndarray, Delta: float, ridge: float = 1e-4, MaxIterations = 10) -> np.ndarray:
        if self.parameters.size == 0:
            self.parameters = np.random.randn(X.shape[1])

        oldParameters = np.zeros_like(self.parameters)

        count = 0

        # iterate until parameter change is below Delta
        while (np.linalg.norm(self.parameters - oldParameters) > Delta and count < MaxIterations):
            count += 1
            oldParameters = self.parameters

        # 1) scores and probabilities
            z = X @ self.parameters
            probs = self.sigmoid(z)

        # 2) gradient of negative log-likelihood
            gradient = X.T @ (probs - Y)

        # 3) diagonal weight matrix
            W = np.diag(probs * (1 - probs))  

        # 4) Hessian and Newton step
            H = X.T @ W @ X
            #add a ridge to ensure invertiblity
            H += ridge * np.eye(H.shape[0])

            step = np.linalg.solve(H, gradient)
            

        # 5) update
            self.parameters -= step

        return self.parameters
            


           