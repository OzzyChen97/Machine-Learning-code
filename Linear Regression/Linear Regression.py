import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_cost(X, y, theta):
    """
    Compute the mean squared error cost J(theta).
    X     : (m × n) design matrix, where first column is all 1s for the intercept.
    y     : (m × 1) target values.
    theta : (n × 1) parameter vector.
    Returns:
        J : scalar cost.
    """
    m = len(y)
    predictions = X.dot(theta)
    sq_errors = (predictions - y) ** 2
    J = (1 / (2 * m)) * np.sum(sq_errors)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Perform gradient descent to learn theta.
    X         : (m × n) design matrix.
    y         : (m × 1) target vector.
    theta     : initial (n × 1) parameter vector.
    alpha     : learning rate.
    num_iters : number of iterations.
    Returns:
        theta     : optimized parameters.
        J_history : list of cost at each iteration.
    """
    m = len(y)
    J_history = []

    for i in range(num_iters):
        error = X.dot(theta) - y            # (m × 1)
        gradient = (1 / m) * (X.T.dot(error))  # (n × 1)
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def plot_cost_function_3d(X, y):
    """
    Visualize the cost function J(theta) in 3D.
    """
    theta_0_vals = np.linspace(-10, 10, 100)  # Values for intercept
    theta_1_vals = np.linspace(-1, 4, 100)    # Values for slope
    cost_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

    # Calculate cost for each combination of theta_0 and theta_1
    for i, theta_0 in enumerate(theta_0_vals):
        for j, theta_1 in enumerate(theta_1_vals):
            theta = np.array([[theta_0], [theta_1]])
            cost_vals[i, j] = compute_cost(X, y, theta)

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

def plot_linear_fit(X_raw, y, theta_opt):
    """
    Plot the linear regression result.
    """
    plt.figure()
    plt.scatter(X_raw, y, label="Training data")
    x_vals = np.linspace(0, 2, 100).reshape(100, 1)
    X_plot = np.hstack([np.ones((100, 1)), x_vals])
    y_vals = X_plot.dot(theta_opt)
    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label="Linear fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Result")
    plt.legend()

def plot_cost_convergence(J_history, iterations):
    """
    Plot the cost convergence during gradient descent.
    """
    plt.figure()
    plt.plot(range(iterations), J_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J")
    plt.title("Gradient Descent Convergence")

if __name__ == "__main__":
    # 1. Generate synthetic data: y = 4 + 3x + noise
    m = 100
    X_raw = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X_raw + np.random.randn(m, 1) * 0.5

    # 2. Prepare X matrix by adding a column of 1s (intercept term)
    X = np.hstack([np.ones((m, 1)), X_raw])  # shape: (m, 2)

    # 3. Initialize fitting parameters
    theta_init = np.zeros((2, 1))
    learning_rate = 0.1
    iterations = 200

    # 4. Run gradient descent
    theta_opt, J_history = gradient_descent(X, y, theta_init, learning_rate, iterations)
    print(f"Learned parameters: intercept = {theta_opt[0,0]:.3f}, slope = {theta_opt[1,0]:.3f}")

    # 5. Plot the linear fit
    plot_linear_fit(X_raw, y, theta_opt)

    # 6. Plot cost convergence
    plot_cost_convergence(J_history, iterations)

    # 7. Plot 3D cost function
    plot_cost_function_3d(X, y)

    # Show all plots
    plt.show()
