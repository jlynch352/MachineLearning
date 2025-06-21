import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from MachineLearningScripts.Perceptron import Perceptron
from MachineLearningScripts.PerformanceCalculator import PerformanceCalculator

# 1) Load data & take the first two features
X, y = load_breast_cancer(return_X_y=True)
X2 = X[:, :2]  # use feature 0 and feature 1

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

# 3) Train Perceptron
model = Perceptron()
model.train(
    X_train,
    y_train,
    LearningRate=0.1,
    Epochs=100
)

# 4) Evaluate on test set
y_pred = model.predict(X_test)
_, acc = PerformanceCalculator(y_test, y_pred).totalCorrect()
print(f"Perceptron test accuracy: {acc*100:.2f}%")

# 5) Extract learned parameters
b, w1, w2 = model.parameters
print("Learned parameters:", model.parameters)

# 6) Plot the data and decision boundary
plt.figure(figsize=(8,6))

# scatter test points
plt.scatter(
    X_test[:,0], X_test[:,1],
    c=y_test, cmap="bwr", marker="x", s=80,
    label="Test"
)

# build a line of x1 values spanning the feature range
x1_min, x1_max = X2[:,0].min() - 1, X2[:,0].max() + 1
xx = np.linspace(x1_min, x1_max, 200)
# compute corresponding x2 on the boundary
yy = -(b + w1 * xx) / w2

# plot decision boundary
plt.plot(xx, yy, 'k--', linewidth=2, label="Decision boundary")

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("2D Perceptron Decision Boundary (Breast Cancer)")
plt.legend()
plt.grid(True)
plt.show()