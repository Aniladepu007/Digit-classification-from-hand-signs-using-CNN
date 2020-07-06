import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import copy
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

# loading data from output of pre-processing step
X = np.load('./unsegmentedImages_grayScale_400x400.npy')
y = np.load('./labels_700_Images.npy')

imgSize = 64

# resizing original aimage from 400x400 to {8,16,32,64,128} sizes
X1 = []
for i in X:
    i = cv2.resize(np.asarray(i.astype(np.float32)),(imgSize,imgSize))
    X1.append(np.asarray(i.astype(np.float32)))

X1 = np.asarray(X1)

x_train, x_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=50)

x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train, x_test = x_train / 1.0, x_test / 1.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16,kernel_size=3,strides=1,padding="same", activation='relu', input_shape=(imgSize, imgSize, 1)),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(32,kernel_size=3,strides=1,padding="same", activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding="same", activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='selu', kernel_initializer='lecun_normal')
])
# used selu for better results {here, selu out-performed relu}

model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# reshaping the data as per requirements of the model.fit() method
temp = np.asarray([ x_train[i].reshape(x_train[i].shape[0], x_train[i].shape[1], 1) for i in range(len(x_train))])
x_train = copy.deepcopy(temp)
temp1 = np.asarray([y_train[i].reshape(1) for i in range(len(y_train))])
y_train = copy.deepcopy(temp1)

#  training the model
model.fit(x_train, y_train, epochs=50)

# reshaping the data as per requirements of the model.evaluate() method
tmp = np.asarray([ x_test[i].reshape(x_test[i].shape[0], x_test[i].shape[1], 1) for i in range(len(x_test))])
x_test = copy.deepcopy(tmp)
tmp1 = np.asarray([y_test[i].reshape(1) for i in range(len(y_test))])
y_test = copy.deepcopy(tmp1)

model.evaluate(x_test,  y_test, verbose=2)

# len(model.layers)

tf.keras.models.save_model(model, './trainedModels/segmentedInput/')
# tf.keras.models.save_model(model, './trainedModels/unsegmentedInput/')

# mm = tf.keras.models.load_model('./trainedModels/unsegmentedInput/')
# mm.summary()
