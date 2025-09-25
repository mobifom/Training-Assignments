"""
ALGORITHM: Logistic Regression Training

INPUT: 
    X: training features [m × n matrix]
    y: training labels [m × 1 vector]
    α: learning rate
    epochs: maximum iterations

OUTPUT:
    w: learned weights [n × 1 vector]
    b: learned bias [scalar]

BEGIN:
    1. Initialize w = zeros(n), b = 0
    2. 
    3. FOR epoch = 1 TO epochs:
    4.     # Forward propagation
    5.     z = X · w + b                    # [m × 1]
    6.     h = sigmoid(z) = 1/(1 + exp(-z)) # [m × 1]
    7.     
    8.     # Compute cost
    9.     cost = -mean(y·log(h) + (1-y)·log(1-h))
    10.    
    11.    # Backward propagation (compute gradients)
    12.    dw = (1/m) · X^T · (h - y)       # [n × 1]
    13.    db = (1/m) · sum(h - y)          # scalar
    14.    
    15.    # Update parameters
    16.    w = w - α · dw
    17.    b = b - α · db
    18.    
    19.    # Optional: check convergence
    20.    IF cost_change < tolerance: BREAK
    21. END FOR
    22.
    23. RETURN w, b
END
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic dataset (binary classification)
X, y = make_classification(
    n_samples=200, n_features=2, n_classes=2,
    n_informative=2, n_redundant=0, random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Model Parameters:")
print(f"Weights (w₁, w₂): {clf.coef_[0]}")
print(f"Bias (b): {clf.intercept_[0]}")
# Predictions
y_pred = clf.predict(X_test)

# Evaluate performance
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot decision boundary
import numpy as np
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("Logistic Regression Classification")
plt.show()
