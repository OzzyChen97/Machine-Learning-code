import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Contains four features: sepal length, sepal width, petal length, petal width
y = iris.target  # Class labels: 0, 1, 2 correspond to three flower species

# 2. Data preprocessing: standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Convert labels to One-hot encoding
y_onehot = np.zeros((y.size, 3))  # 3 classes
y_onehot[np.arange(y.size), y] = 1

# 4. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=42)

# 5. Linear regression fitting: compute weights using the normal equation
def fit_linear_regression(X, y):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Compute weights using the normal equation: w = (X^T X)^-1 X^T y
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

# 6. Train multiple models (One-vs-Rest)
weights = []
for i in range(3):  # 3 classes
    model_weights = fit_linear_regression(X_train, y_train[:, i])
    weights.append(model_weights)

weights = np.array(weights)

# 7. Prediction: use the model to predict (for each sample, choose the class with the highest score)
def predict(X, weights):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    scores = X_b.dot(weights.T)  # Compute scores for each class
    return np.argmax(scores, axis=1)  # Return the class with the highest score

# 8. Predict and evaluate the model
y_pred = predict(X_test, weights)
y_true = np.argmax(y_test, axis=1)  # Convert One-hot encoding back to labels
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot 6 decision boundary plots for different feature combinations

# Feature combinations: (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
feature_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Adjust figure size and reduce space between subplots
plt.figure(figsize=(12, 8))  # Reduce the size for better spacing

for i, (f1, f2) in enumerate(feature_combinations):
    # Select features
    X_pair = X_scaled[:, [f1, f2]]

    # Plot decision boundary
    plt.subplot(2, 3, i + 1)

    # Plot decision boundary
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Create a zero array for unused features
    test_points = np.zeros((xx.ravel().shape[0], 4))
    test_points[:, [f1, f2]] = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(test_points, weights)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, edgecolors='k', marker='o')

    # Set title
    plt.title(f'Features {f1 + 1} and {f2 + 1}')
    plt.xlabel(f'Feature {f1 + 1}')
    plt.ylabel(f'Feature {f2 + 1}')

# Adjust layout to avoid overlapping text
plt.tight_layout()
plt.show()
