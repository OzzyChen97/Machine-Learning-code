import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载 Iris 数据集
iris = datasets.load_iris()
X = iris.data  # 包含四个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target  # 类别标签：0, 1, 2 对应三种花

# 2. 数据预处理：标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 使用逻辑回归进行多类分类
model = LogisticRegression(multi_class='ovr', solver='liblinear')  # 'ovr'表示一对多策略
model.fit(X_train, y_train)

# 5. 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 绘制6个不同特征组合的决策边界图

# 特征组合: (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
feature_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# 调整图像大小，并减少子图之间的间距
plt.figure(figsize=(12, 8))  # 适当减小尺寸

for i, (f1, f2) in enumerate(feature_combinations):
    # 选择特征
    X_pair = X_scaled[:, [f1, f2]]

    # 训练逻辑回归模型
    model.fit(X_train[:, [f1, f2]], y_train)

    # 绘制决策边界
    plt.subplot(2, 3, i + 1)

    # 绘制决策边界
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, edgecolors='k', marker='o')

    # 设置标题
    plt.title(f'Features {f1 + 1} and {f2 + 1}')
    plt.xlabel(f'Feature {f1 + 1}')
    plt.ylabel(f'Feature {f2 + 1}')

# 调整布局以避免文字重叠
plt.tight_layout()

# 如果需要更精细的调整，可以使用以下方法：
# plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()
