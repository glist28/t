import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

n = 10
fix, axs = plt.subplots(ncols=n)
for i in range(n):
    axs[i].imshow(x_train[i], cmap='gray')
    axs[i].axis('off')
print(x_train.shape[0], 'изображений для обучения')
print(x_train.test[0], 'изображений для обучения')
print('начальный размер x_train shape:', x_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_train.reshape(-1, 28, 28, 1)
print('Преобразованный размер x_train shape: ', x_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (28, 28, 1)
model = keras.Sequential(
    [
        keras.Input(shape = input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ]
)
print(model.summary())
epochs = 5
batch_size = 128
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test)

