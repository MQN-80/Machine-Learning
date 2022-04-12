import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")
features=pd.read_csv('C:\\users\\86155\\Desktop\\temps0.csv')
features=pd.get_dummies(features)
print(features.head(5))
labels=np.array(features['actual'])
features=features.drop('actual',axis=1)
feature_list=list(features.columns)
features=np.array(features)
print(features.shape)
input_features=preprocessing.StandardScaler().fit_transform(features)
model=tf.keras.Sequential()
model.add(layers.Dense(16,kernel_initializer='random_normal'))
model.add(layers.Dense(32,kernel_initializer='random_normal'))
model.add(layers.Dense(1,kernel_initializer='random_normal'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
              loss='mean_squared_error')
model.fit(input_features,labels,validation_split=0.25,epochs=100,batch_size=64)