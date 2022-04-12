import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# 用于基本keras的学习
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
k22=Kera()
k22.k3()
