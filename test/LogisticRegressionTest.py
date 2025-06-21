import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MachineLearningScripts.LogisticRegression import LogisticRegression
from MachineLearningScripts.PerformanceCalculator import PerformanceCalculator

# 1) Load data & pick feature 0
X, Y = load_breast_cancer(return_X_y=True)
X0 = X[:, 0].reshape(-1, 1)

# 2) Standardize
scaler = StandardScaler()
X0_scaled = scaler.fit_transform(X0)

# 3) Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X0_scaled, Y, test_size=0.2, random_state=42
)

# 4) Fit with Newton’s method
model = LogisticRegression(threshold=0.5)
model.NewtonsMethod(
    X_train,
    Y_train,
    Delta=1e-6,
    ridge=1e-4,
    MaxIterations=50
)

# 5) Prepare a smooth grid over the feature’s range
x_axis = np.linspace(
    X0_scaled.min(), 
    X0_scaled.max(), 
    200
).reshape(-1, 1)

# 6) Compute predicted probabilities
y_prob = model.PredictionProbability(x_axis)
y_pred = model.PredictionClass(X_test)

calc = PerformanceCalculator(Y_test,y_pred)
_, PercentCorrect = calc.totalCorrect()

print(PercentCorrect)

# 7) Plot
plt.figure(figsize=(8,5))

# true classes
plt.scatter(
    X_test[:,0], Y_test, 
    c=Y_test, cmap="bwr", 
    edgecolor="k", alpha=0.7, 
    label="True Class"
)
# logistic curve
plt.plot(
    x_axis, y_prob, 
    linewidth=2, label="Logistic Curve"
)
# threshold line
plt.axhline(0.5, color="gray", linestyle="--", label="Threshold = 0.5")

plt.xlabel("Standardized Feature 0")
plt.ylabel("Predicted Probability")
plt.title("Logistic Regression Fit (Newton’s Method)")
plt.legend()
plt.grid(True)
plt.show()





