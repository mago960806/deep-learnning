import time

from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

start = time.time()

(train_images, train_lables), (test_images, test_lables) = mnist.load_data()

print(f"训练集共有: {len(train_images)}")
print(f"测试集共有: {len(test_images)}")

# 创建一个神经网络
network = models.Sequential()
# 在神经网络中添加两层全连接神经层(Dense)
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

# 编译模型
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 处理图像数据
# 三维张量转二维张量并修改数据类型为float32
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 准备标签
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)

# 开始训练
network.fit(train_images, train_lables, epochs=5, batch_size=128)
end = time.time()
print(f"模型训练完毕, 耗时: {(end - start):.3}s")
# 测试模型
test_loss, test_acc = network.evaluate(test_images, test_lables)
print(f"测试集精准度为: {test_acc:.2%}")
