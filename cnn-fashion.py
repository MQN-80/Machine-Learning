import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers, optimizers
fashion_mnist =keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
fig = plt.figure(figsize=[10, 8])
for i in range(6): #可视化部分数据
  ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
  ax.imshow(x_train[i])
  plt.imshow(x_train[i])
  ax.set_title(str(y_train[i]))
plt.show()
