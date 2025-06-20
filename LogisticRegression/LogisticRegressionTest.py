from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# import your modules
from LogisticRegression.LogisticRegression import LogisticRegression
from HelperClasses.PerformanceCalculator import PerformanceCalculator

# 1) Load data
X, Y = load_breast_cancer(return_X_y=True)

# 2) For visualization, we’re only using feature 0 + intercept
X0 = X[:, 0].reshape(-1, 1)

# 3) Standardize the feature (helps Newton converge faster)
scaler = StandardScaler()
X0_scaled = scaler.fit_transform(X0)

# 4) Add intercept column
X_all = np.hstack([np.ones((X0_scaled.shape[0], 1)), X0_scaled])

# 5) Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y, test_size=0.2, random_state=42
)

# 6) Instantiate and fit with Newton’s method
model = LogisticRegression(threshold=0.5)
model.NewtonsMethod(
    X_train, 
    Y_train, 
    Delta=1e-6,      
    ridge=1e-4,      
    MaxIterations=50      
)

# 7) Make predictions on the test set
predictedClasses = model.PredictionClass(X_test)

# 8) Compute accuracy
calc = PerformanceCalculator(predictedClasses, Y_test)
_, accuracy = calc.totalCorrect()
print(f"Newton’s method test accuracy: {accuracy*100:.2f}%")

# 9) Plot the fit and the data
plt.figure(figsize=(8,5))

# logistic curve
x_axis = np.linspace(X_test[:,1].min(), X_test[:,1].max(), 200).reshape(-1,1)
X_curve = np.hstack([np.ones_like(x_axis), x_axis])
y_curve = model.sigmoid(X_curve @ model.parameters)
plt.plot(x_axis, y_curve, label="Newton Fit", linewidth=2)

# threshold line
plt.hlines(model.threshold, x_axis.min(), x_axis.max(),
           linestyles="--", label=f"Threshold = {model.threshold}")

# scatter actual vs predicted
plt.scatter(X_test[:,1], Y_test, c=Y_test, cmap="bwr",
            edgecolor="k", alpha=0.7, label="True Class")
plt.scatter(X_test[:,1], predictedClasses, marker="x", c="green",
            label="Predicted Class")

plt.xlabel("Standardized Feature 0")
plt.ylabel("Probability / Class")
plt.title("Logistic Regression (Newton) on Breast-Cancer Data")
plt.legend()
plt.show()








