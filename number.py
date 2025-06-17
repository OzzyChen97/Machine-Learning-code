import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# --------------------------
# 1. 数据加载与预处理
# --------------------------

# 读取 CSV
train_df = pd.read_csv('/Users/ozzychen/Desktop/Machine Learning code/Ozzy\'s Machine Learning/data/sign_mnist_train/sign_mnist_train.csv')
test_df  = pd.read_csv('/Users/ozzychen/Desktop/Machine Learning code/Ozzy\'s Machine Learning/data/sign_mnist_test/sign_mnist_test.csv')

# 分离标签与像素值
y_train = train_df.pop('label').values
X_train = train_df.values
y_test  = test_df.pop('label').values
X_test  = test_df.values

# 确保标签值在 [0, 9] 范围内（处理超出范围的标签值）
y_train = np.clip(y_train, 0, 9)
y_test  = np.clip(y_test, 0, 9)

# 将一维向量转换为 (28,28,1)，并归一化到 [0, 1] 区间
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 转换标签为 one-hot 编码格式
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# --------------------------
# 2. 构建 CNN 模型
# --------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')   # 0–9 共 10 类
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------
# 3. 训练模型
# --------------------------

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# --------------------------
# 4. 测试模型
# --------------------------

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'[测试集] 准确率: {acc*100:.2f}%')

# --------------------------
# 5. 实时手势数字识别
# --------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 抠出手势区域：这里简单取中央 200×200 区域，可根据实际情况改进
    h, w = frame.shape[:2]
    x0, y0 = w//2 - 100, h//2 - 100
    roi = frame[y0:y0+200, x0:x0+200]

    # 预处理
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th  = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    digit  = cv2.resize(th, (28,28), interpolation=cv2.INTER_AREA)
    digit  = digit.astype('float32') / 255.0
    digit  = np.expand_dims(digit, -1)   # → (28,28,1)
    digit  = np.expand_dims(digit,  0)   # → (1,28,28,1)

    # 预测
    pred = model.predict(digit, verbose=0)
    cls  = np.argmax(pred)

    # 绘制结果
    cv2.rectangle(frame, (x0,y0), (x0+200,y0+200), (0,255,0), 2)
    cv2.putText(frame, f'Pred: {cls}', (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow('Hand Gesture Digit Recognizer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
