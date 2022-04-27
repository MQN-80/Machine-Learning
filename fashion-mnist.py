import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#采用多种模型对fashion-mnist数据集进行学习，并比较准确性损失
class Model:
    fashion_mnist =keras.datasets.fashion_mnist
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
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # 载入数据
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

    def image_plot(self,i, pred, true_label, img):
        labels = ['T-shirt/top',
                  'Trouser',
                  'Pullover',
                  'Dress',
                  'Coat',
                  'Sandal',
                  'Shirt',
                  'Sneaker',
                  'Bag',
                  'Ankle boot']
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        pred_label = np.argmax(pred)
        if pred_label == true_label:
            color = 'green'
        else:
            color = 'red'
        plt.xlabel("Predicted: {} \nProb: {:2.0f}% \n(Actual: {})".format(labels[pred_label],
                                                                          100 * np.max(pred),
                                                                          labels[true_label]),
                   color=color)
    def a3(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()  # 载入数据
        x_train = x_train.reshape(x_train.shape[0],28,28,1)/ 255.0
        x_test = x_test.reshape(x_test.shape[0],28,28,1) / 255.0
        # 换为one hot格式
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        model = keras.Sequential([
            keras.layers.Conv2D(filters=6, kernel_size=5,
                                strides=1, activation='relu',
                                input_shape=(28, 28, 1), padding='same'),

            keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),

            keras.layers.Conv2D(filters=16, kernel_size=5,
                                strides=1, activation='relu', padding='valid'),
            keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),

            keras.layers.Conv2D(filters=120, kernel_size=5,
                                strides=1, activation='relu', padding='valid'),

            keras.layers.Flatten(),
            keras.layers.Dense(units=84, activation='relu'),
            #keras.layers.Dropout(0.2),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        sgd = keras.optimizers.SGD(lr=0.2)  # 定义优化器，梯度下降算法
        model.compile(
            optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.summary()
        # 训练模型
        hist=model.fit(x_train, y_train, batch_size=32, epochs=10,validation_data=(x_test, y_test),callbacks=[])
        #model.save('lstm_model.h5')
        f, ax = plt.subplots(1, 2, figsize=[18, 8])
        ax[0].plot([None] + hist.history['acc'], 'o-')
        ax[0].plot([None] + hist.history['val_acc'], 'x-')
        # Plot legend and use the best location automatically: loc = 0.
        ax[0].legend(['Train acc', 'Validation acc'], loc=0)
        ax[0].set_title('Training/Validation acc per Epoch')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('acc')

        ax[1].plot([None] + hist.history['loss'], 'o-')
        ax[1].plot([None] + hist.history['val_loss'], 'x-')

        # Plot legend and use the best location automatically: loc = 0.
        ax[1].legend(['Train loss', "Val loss"], loc=0)
        ax[1].set_title('Training/Validation Loss per Epoch')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')

        plt.tight_layout()
        plt.show()
        # 评估模型
        loss, accuracy = model.evaluate(x_test, y_test)
    def a5(self):
        model1 = keras.models.load_model('lstm_model.h5')
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()  # 载入数据
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0
        # 换为one hot格式
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        pred = model1.predict(x_test)
        loss, accuracy = model1.evaluate(x_test, y_test)

        print('\ntest loss', loss)
        print('accuracy', accuracy)
        y = np.argmax(y_test, axis=-1)
        fig = plt.figure(figsize=[20, 15])
        for i in range(10):
            ax = fig.add_subplot(1, 10,i+1, xticks=[], yticks=[])
            self.image_plot(i,pred[i],y,x_test)
        plt.show()
example = Model()
example.a5()
