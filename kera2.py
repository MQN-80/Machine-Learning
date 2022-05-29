import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2 as cv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class Kera:
    def k1(self): #对于具有两个类的单输入模型
        model=keras.models.Sequential()#设置训练模型,顺序模型是多个网络层的线性堆叠
        model.add(keras.layers.Dense(64,activation='relu',input_dim=100)) #添加一个有64个单元全连接层到模型
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        data=np.random.random((1000,100))
        labels=np.random.randint(2,size=(1000,1))
        model.fit(data,labels,epochs=10,batch_size=32)
        print(model.get_config())
    def k2(self): #实战线性回归
        x_data=np.random.rand(100) #生成0到1随机数
        noise=np.random.normal(0,0.01,x_data.shape) #正态分布噪声
        y_data=x_data*0.1+0.2+noise
        plt.scatter(x_data,y_data)
        #建立线性模型部分
        model=keras.Sequential()
        model.add(keras.layers.Dense(units=1,input_dim=1))
        model.compile(optimizer='sgd',loss='mse')
        model.fit(x_data,y_data,epochs=1000,batch_size=36)
        y_pred=model.predict(x_data)
        plt.plot(x_data,y_pred,'r-',lw=3)
        plt.show()
    def k3(self): #非线性回归预测模型
        x_data=np.linspace(-0.5,0.5,200)
        noise=np.random.normal(0,0.02,x_data.shape)
        y_data=np.square(x_data)+noise
        plt.scatter(x_data,y_data)
        model=keras.Sequential()
        model.add(keras.layers.Dense(units=10,input_dim=1,activation='relu'))
        model.add(keras.layers.Dense(units=1,activation='relu'))
        sgd=keras.optimizers.SGD(lr=0.3)
        model.compile(optimizer=sgd, loss='mse')
        model.fit(x_data,y_data,epochs=500,batch_size=36) #训练次数不能太多，防止过拟合现象的发生
        y_pred=model.predict(x_data)
        plt.plot(x_data,y_pred,'r-',lw=3)
        plt.show()
    def k4(self):  #交叉熵
        (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()  #载入数据
        print('x_shape:',x_train.shape)
        print('y_shape',y_train.shape)
        x_train=x_train.reshape(x_train.shape[0],-1)/255.0
        x_test=x_test.reshape(x_test.shape[0],-1)/255.0
        # 换为one hot格式
        y_train=keras.utils.to_categorical(y_train,num_classes=10)
        y_test=keras.utils.to_categorical(y_test,num_classes=10)
        #创建模型，输入784个神经元，输出10个神经元
        model=keras.Sequential([
            keras.layers.Dense(units=10,input_dim=784,bias_initializer='one',activation='softmax')
        ])
        sgd=keras.optimizers.SGD(lr=0.2)  #定义优化器，梯度下降算法
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
    # k5:dropout层的加入，防止过拟合，Dropout(0.3)加入丢弃率为0.3的层
    # k6(self):  #引入正则化惩罚，kernel_regularizer=l2(0.0003)
    def k7(self): #可视化显示
        def image_plot(i, pred, true_label, img):
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
    def k8(self):
        import tensorflow_hub as hub

        # Load the input image.
        image_path = "E:\\下载\\human-pose-estimation-opencv-master\\1.jpeg"
        image = tf.io.read_file(image_path)
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

        # Download the model from TF Hub.
        model = hub.load("movenet_multipose_lightning_1")
        movenet = model.signatures['serving_default']

        # Run model inference.
        outputs = movenet(image)
        # Output is a [1, 6, 56] tensor.
        keypoints = outputs['output_0']
        print(keypoints.numpy())
    def k9(self): #tensor基本操作
        rank_0_tensor = tf.constant(4)
        print(rank_0_tensor)
k22=Kera()
k22.k2()
