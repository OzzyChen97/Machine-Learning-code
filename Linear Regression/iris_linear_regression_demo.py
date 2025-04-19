import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_mse_cost(design_matrix, targets, parameters):
    m = len(targets)
    predictions = design_matrix.dot(parameters)
    squared_errors = (predictions - targets) ** 2
    return (1 / (2 * m)) * np.sum(squared_errors)

def gradient_descent(design_matrix, targets, initial_params, learning_rate, num_iterations):
    m = len(targets)
    parameters = initial_params.copy()
    cost_history = []
    
    for _ in range(num_iterations):
        error = design_matrix.dot(parameters) - targets
        gradient = (1 / m) * design_matrix.T.dot(error)
        parameters -= learning_rate * gradient
        cost_history.append(compute_mse_cost(design_matrix, targets, parameters))
    
    return parameters, cost_history

def plot_cost_function_3d(X, y):
    theta_0_vals = np.linspace(-10, 10, 100)  # Values for intercept
    theta_1_vals = np.linspace(-1, 4, 100)    # Values for slope
    cost_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

    # Calculate cost for each combination of theta_0 and theta_1
    for i, theta_0 in enumerate(theta_0_vals):
        for j, theta_1 in enumerate(theta_1_vals):
            theta = np.array([[theta_0], [theta_1]])
            cost_vals[i, j] = compute_mse_cost(X, y, theta)

    # Create meshgrid for plotting
    theta_0_vals, theta_1_vals = np.meshgrid(theta_0_vals, theta_1_vals)

    # Plotting the 3D cost function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta_0_vals, theta_1_vals, cost_vals, cmap='viridis')
    ax.set_xlabel('Intercept (theta_0)')
    ax.set_ylabel('Slope (theta_1)')
    ax.set_zlabel('Cost (J)')
    ax.set_title('3D Visualization of the Cost Function')
    plt.show()

def main():
    # 1. Load Iris dataset from URL
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    data = pd.read_csv(url)

    # 2. Select feature (petal length) and target (sepal length)
    petal_length = data["petal_length"].values.reshape(-1, 1)
    sepal_length = data["sepal_length"].values.reshape(-1, 1)

    # 3. Prepare design matrix by adding intercept term
    m = len(sepal_length)
    intercept_column = np.ones((m, 1))
    X = np.hstack([intercept_column, petal_length])  # shape (m, 2)

    # 4. Initialize parameters and hyperparameters
    initial_theta = np.zeros((2, 1))  # [intercept, slope]
    alpha = 0.01                      # learning rate
    iterations = 1000                 # number of gradient steps

    # 5. Run gradient descent
    learned_theta, cost_history = gradient_descent(X, sepal_length, initial_theta, alpha, iterations)
    intercept, slope = learned_theta.ravel()
    print(f"Learned model parameters: intercept = {intercept:.3f}, slope = {slope:.3f}")

    # 6. Plot the linear fit
    plt.figure()
    plt.scatter(petal_length, sepal_length, label="Data points")
    x_vals = np.linspace(petal_length.min(), petal_length.max(), 100).reshape(-1, 1)
    X_plot = np.hstack([np.ones((100, 1)), x_vals])
    y_vals = X_plot.dot(learned_theta)
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label="Linear fit")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Sepal Length (cm)")
    plt.title("Iris Dataset: Linear Regression")
    plt.legend()

    # 7. Plot cost function convergence
    plt.figure()
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J")
    plt.title("Gradient Descent Convergence")

    # 8. Plot 3D cost function
    plot_cost_function_3d(X, sepal_length)

    plt.show()

if __name__ == "__main__":
    main()