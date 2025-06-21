import numpy as np

class Perceptron:

    def __init__(self):
        self.parameters: np.ndarray = np.array([])

    def classify(self, Z: np.ndarray) -> np.ndarray:

        return  (Z >= 0).astype(int)

    def train(self,X:np.ndarray, Y: np.ndarray, LearningRate: float = 0.01, Epochs:int = 10) -> np.ndarray:

        m, n = X.shape
        X1 = np.column_stack([np.ones((m,1)), X])

        if self.parameters.size == 0:
            self.parameters = np.random.randn(n +1 )

        for _ in range(Epochs):
            errors = 0

            for x,y in zip(X1,Y):
                
                z = x @ self.parameters

                prediction = self.classify(z)

                update = LearningRate * (y - prediction)

                if update != 0:
                    self.parameters += update * x
                    errors += 1

            if errors == 0:
                break
            
        return self.parameters

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.parameters.size == 0:
            print("You haven't trained the model yet")
            return np.ndarray([])
        
        m, n = X.shape
        
        X1 = np.column_stack([np.ones((m,1)), X])
        z = X1 @ self.parameters
        return self.classify(z)
        






        