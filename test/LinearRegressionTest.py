import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

#my modules
from MachineLearningScripts.LinearRegression import LinearRegression
from MachineLearningScripts.PerformanceCalculator import PerformanceCalculator

# load raw data
data = load_diabetes()
X_raw, Y = data.data, data.target

varibale = 3 
X1 = X_raw[:,varibale]

# augment with a column of ones for the intercept term

OLS = LinearRegression()
OLS.OrdinaryLeastSquaresMethod(X1,Y)
predictionsOLS = OLS.Prediction(X1)

SGD = LinearRegression()
SGD.StochasticGradientDescent(X1,Y,LearningRate=0.01,Epochs=500)
predictionsSGD = SGD.Prediction(X1)

BGD = LinearRegression()
BGD.BatchGradientDescent(X1,Y,LearningRate=0.1,Epochs=5000)
predictionsBGD = BGD.Prediction(X1)

SBGD = LinearRegression()
SBGD.StochasticBatchGradientDescent(X1,Y,LearningRate=0.1,Epochs=5000,batchSize=10)
predictionsSBGD = SBGD.Prediction(X1)


##MultiLinearRegression
X2 = X_raw

multi = LinearRegression()
multi.OrdinaryLeastSquaresMethod(X2, Y)
predictionsMulti = multi.Prediction(X2)

multiRegressionEval = PerformanceCalculator(Y, predictionsMulti)

mseM = multiRegressionEval.mse()
rmseM = multiRegressionEval.rmse()
maeM = multiRegressionEval.mae()
r2M = multiRegressionEval.r2()

print(f"Multi Regression - mse: {mseM}, rmse: {rmseM}, mae: {maeM}, r^2: {r2M}")

## comparing multi regression against a single linear regression
OLSRegressionEval = PerformanceCalculator(Y, predictionsOLS)

mseOLS = OLSRegressionEval.mse()
rmseOLS = OLSRegressionEval.rmse()
maeOLS = OLSRegressionEval.mae()
r2OLS = OLSRegressionEval.r2()

print(f"OLS Regression - mse: {mseOLS}, rmse: {rmseOLS}, mae: {maeOLS}, r^2: {r2OLS}")

# Testing Locallaly Weighted Regresssion against sin(X)
llr = LinearRegression()

x = np.linspace(0, 10, 100)
y = 6*np.sin(x) + 1 + np.random.randn(100) * 2
X = np.column_stack((np.ones_like(x), x))

# Choose bandwidth
tau = 0.5

# Predict at test points
x_test = np.linspace(0, 10, 100)
y_pred = []
for xi in x_test:
    theta = llr.LocallyWeightedRegression(X, y, np.array([1, xi]), tau)
    y_pred.append(theta[0] + theta[1]*xi)
y_pred = np.array(y_pred)

# First subplot: diabetes regressions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X1, Y, label="Data", alpha=0.6)
ax1.plot(X1, predictionsOLS,  label="OLS",  linewidth=2, color="red")
ax1.plot(X1, predictionsSGD,  label="SGD",  linewidth=2, color="green")
ax1.plot(X1, predictionsBGD,  label="BGD",  linewidth=2, color="yellow")
ax1.plot(X1, predictionsSBGD, label="SBGD", linewidth=2, color="orange")

ax1.set_xlabel("Independent Var")
ax1.set_ylabel("Target")
ax1.set_title("Linear Regression Methods")
ax1.legend()
ax1.grid(True)

# Second subplot: LWR on nonlinear data
ax2.scatter(x, y, alpha=0.6)
ax2.plot(x_test, y_pred, linewidth=2)

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Locally Weighted Regression")
ax2.grid(True)

plt.show()