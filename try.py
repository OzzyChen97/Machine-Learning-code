import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义目标函数 f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# 计算目标函数的梯度（即偏导数）
def df(x, y):
    df_dx = 2 * x  # 对 x 的偏导数
    df_dy = 2 * y  # 对 y 的偏导数
    return np.array([df_dx, df_dy])

# 梯度下降算法
def gradient_descent(starting_point, learning_rate, num_iterations):
    x, y = starting_point
    trajectory = [starting_point]  # 用来记录每次迭代的点
    for _ in range(num_iterations):
        grad = df(x, y)  # 计算梯度
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]  # 更新参数
        trajectory.append([x, y])  # 存储每次迭代的点
    return np.array([x, y]), np.array(trajectory)

# 可视化梯度下降过程
def plot_function_and_path():
    # 创建 x 和 y 的网格
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    # 绘制目标函数的三维曲面
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # 设置初始点、学习率和迭代次数
    starting_point = np.array([1.5, 1.5])  # 起始点
    learning_rate = 0.1  # 学习率
    num_iterations = 50  # 迭代次数
    
    # 执行梯度下降
    final_point, trajectory = gradient_descent(starting_point, learning_rate, num_iterations)

    # 绘制梯度下降路径
    ax.plot(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]), 'r-', marker='o', markersize=5, label="Gradient Descent Path")
    ax.scatter(final_point[0], final_point[1], f(final_point[0], final_point[1]), color='blue', label="Final Point", zorder=5)

    # 绘图设置
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Gradient Descent on f(x, y) = x^2 + y^2')
    ax.legend()
    plt.show()

# 执行绘制和梯度下降
plot_function_and_path()
