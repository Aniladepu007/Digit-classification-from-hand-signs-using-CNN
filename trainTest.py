import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import copy
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

# loading data from output of preProcessing_numpyArrayFormat.py step
X = np.load('./unsegmentedImages_grayScale_400x400.npy')
y = np.load('./labels_700_Images.npy')

imgSize = 32

# resizing original images from 400x400 to different{8,16,32,64,128} sizes
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
# print(len(model.layers))


# reshaping the data as per requirements of the model.fit() method
temp = np.asarray([ x_train[i].reshape(x_train[i].shape[0], x_train[i].shape[1], 1) for i in range(len(x_train))])
x_train = copy.deepcopy(temp)
temp1 = np.asarray([y_train[i].reshape(1) for i in range(len(y_train))])
y_train = copy.deepcopy(temp1)

# reshaping the data as per requirements of the model.evaluate() method
tmp = np.asarray([ x_test[i].reshape(x_test[i].shape[0], x_test[i].shape[1], 1) for i in range(len(x_test))])
x_test = copy.deepcopy(tmp)
tmp1 = np.asarray([y_test[i].reshape(1) for i in range(len(y_test))])
y_test = copy.deepcopy(tmp1)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#  training the model
history = model.fit(x_train,
                    y_train,
                    epochs = 50,
                    batch_size = 4,
                    validation_split = 0.1,
                    validation_freq = 2,
                    verbose=2
                )

#testing the model
model.evaluate(x_test,  y_test, verbose=2)

predicted = model.predict(x_test, verbose=2)
plt.imshow(x_test[0].reshape(x_test[0].shape[0], x_test[0].shape[1]), cmap='gray')
plt.show()

print(predicted[0])


tf.keras.models.save_model(model, './trainedModels/segmentedInput/')
# tf.keras.models.save_model(model, './trainedModels/unsegmentedInput/')

# mm = tf.keras.models.load_model('./trainedModels/unsegmentedInput/')
# mm.summary()

plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss')
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)
plt.show()

# tf.keras.models.save_model(model, './trainedModels/segmentedInput/')
# tf.keras.models.save_model(model, './trainedModels/unsegmentedInput/')

# preTrainedModel = tf.keras.models.load_model('./trainedModels/unsegmentedInput/')
# preTrainedModel.summary()
