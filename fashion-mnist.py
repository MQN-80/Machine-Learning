import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#采用多种模型对fashion-mnist数据集进行学习，并比较准确性损失
class Model:
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # 导入fashion_mnist数据，前两个数组为训练集，为后续模型学习构建学习数据，后两个数据为测试集，对模型进行测试
    def a1(self): #第一种，采用adam优化器，损失函数使用sparse_categorical_crossentropy
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # 该函数用于将图像格式从二维数组转为一维数组，以格式化数据，便于神经层的输入
            keras.layers.Dense(128, activation='relu'),  # 第一个神经层，共128个节点
            keras.layers.Dense(10)  # 第二个神经层，10个神经元，以表示当前图像属于10个类中的哪一类
        ])
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])  # 模型的编译步骤，设置合适的损失函数，优化器和指标
        model.fit(self.train_images, self.train_labels, epochs=10)  # 进行模型的训练
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)  # 验证训练模型在测试集上的准确性
        print('\nTest accuracy:', test_acc)
    def a2(self): #先转换为one hot格式，优化器为SGD梯度下降，损失函数为categorical_crossentropy
        (x_train, y_train), (x_test, y_test) = self.fashion_mnist.load_data()  # 载入数据
        print('x_shape:', x_train.shape)
        print('y_shape', y_train.shape)
        x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
        # 换为one hot格式
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # 创建模型，输入784个神经元，输出10个神经元
        model = keras.Sequential([
            keras.layers.Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
        ])
        sgd = keras.optimizers.SGD(lr=0.2)  # 定义优化器，梯度下降算法
        model.compile(
            optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # 训练模型
        model.fit(x_train, y_train, batch_size=32, epochs=10)

        # 评估模型
        loss, accuracy = model.evaluate(x_test, y_test)

        print('\ntest loss', loss)
        print('accuracy', accuracy)
example = Model()
example.a2()
