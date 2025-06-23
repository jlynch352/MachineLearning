import numpy as np

''' 
Logistic Regression From Scrath

File Contiains
1. Prediction
2. Stochastic Gradient Descent
5. Newtons Method
'''

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
    
    def PredictionProbability(self, X: np.ndarray) -> np.ndarray:
        if self.parameters.size == 0:
            print("You haven't trained the model yet")
            return np.array([])
        
        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        return self.sigmoid(X1 @ self.parameters)
    
    def PredictionClass(self,X: np.ndarray) -> np.ndarray:

        probs = self.PredictionProbability(X)
        return (probs >= self.threshold).astype(int)
    
    def StochasticGraidentDescent(self, X: np.ndarray, Y: np.ndarray, LearningRate: float, Epochs: int) -> np.ndarray:

        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])

        for _ in range(Epochs):
            
            for x,y in zip(X1,Y):
                graident = (y - self.sigmoid(x @ self.parameters)) * x
                self.parameters = self.parameters + LearningRate * graident

        return self.parameters

    def NewtonsMethod(self, X: np.ndarray, Y: np.ndarray, Delta: float, ridge: float = 1e-4, MaxIterations: int = 10) -> np.ndarray:

        X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(X1.shape[1])

        for _ in range(MaxIterations):

            #Find predicted probability
            z = X1 @ self.parameters
            probs = self.sigmoid(z)

            # calculate the graident 
            gradient = X1.T @ (probs - Y)

            # calculate W for calculating the Hessian
            W = np.diag(probs * (1 - probs))  

            # calculate the hessian
            H = X1.T @ W @ X1

            #add ridge to stop non invertible matrix
            H += ridge * np.eye(H.shape[0])

            # do Graident / Hesian but do via solving the equation
            step = np.linalg.solve(H, gradient)

            #update the paramters

            self.parameters -=  step

            # once accurate enough end
            if np.linalg.norm(step) < Delta:
                return self.parameters

        return self.parameters
            


           