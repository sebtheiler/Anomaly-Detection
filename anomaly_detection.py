from keras.datasets import fashion_mnist
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Conv2DTranspose, Lambda, Reshape
from keras.callbacks import LambdaCallback
from keras.backend import tf # as ktf
import keras.backend as K
import numpy as np
import random, os, math
import matplotlib.pyplot as plt

def load(filepath):
    model = load_model(filepath)
    encoder = Model(model.layers[0].input, model.layers[7].output)
    return model, encoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train, x_train.shape,y_train)

fashion_labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


epochs = 50
batch_size = 64
steps_per_epoch = None
save_model = False
save_to = 'fashion_autoencoder.hdf5'
load_from = None # File to load model from.  None for new model.


if load_from == None:
    conv_input = Input(shape=(None, 28, 28))

    conv_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_input)
    pool_1 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv_1)

    conv_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv_2)

    conv_3 = Conv2D(8, (3, 3), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv_3)

    conv_4 = Conv2D(4, (3, 3), padding='same', activation='relu')(pool_3) # Encoded state
    pool_4 = Lambda(lambda image: K.tf.image.resize_images(image, (7, 7)))(conv_4)

    conv_5 = Conv2D(8, (3, 3), padding='same', activation='relu')(pool_4)
    pool_5 = Lambda(lambda image: K.tf.image.resize_images(image, (14, 14)))(conv_5)

    conv_6 = Conv2D(16, (3, 3), padding='same', activation='relu')(pool_5)
    pool_6 = Lambda(lambda image: K.tf.image.resize_images(image, (28, 28)))(conv_6)

    conv_output_1 = Conv2D(1, (3, 3), padding='same', activation='relu')(pool_6)
    conv_output_2 = Reshape((1, 28, 28), name='conv_output2')(conv_output_1)

    model = Model(conv_input, conv_output_2)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    # encoder = Model(conv_input, conv_4)
    encoder = Model(model.layers[0].input, model.layers[7].output)
    encoder.compile(optimizer='adam', loss='mean_squared_error')
else:
    model, encoder = load(load_from)

x_train = x_train.reshape((60000, 1, 28, 28))
x_test = x_test.reshape((10000, 1, 28, 28))


model.fit(x=x_train,
          y=x_train,
          validation_data=(x_test, x_test),
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          batch_size=batch_size)

if save_model:
    model.save(save_to)

originals = []
encodeds = []
decodeds = []
indexes = []

examples = 10

for _ in range(examples):
    n = random.randint(0, 9999)
    originals.append(x_test[n].reshape((28, 28)))
    encodeds.append(encoder.predict(x_test[n].reshape((1, 1, 28, 28))).reshape((4, 4)))
    decodeds.append(model.predict(x_test[n].reshape((1, 1, 28, 28))).reshape((28, 28)))
    indexes.append(y_test[n])


images = []
for i in range(examples):
    images.append(originals[i])
    images.append(encodeds[i])
    images.append(decodeds[i])

# Ground truth, encoded, and reconstructed images
w = 10
h = 10
fig = plt.figure(figsize=(28, 28))
columns = 3
rows = examples
for counter, i in enumerate(range(1, columns*rows + 1)):
    img = images[counter]
    fig.add_subplot(rows, columns, i).set_title('{label}'.format(label=fashion_labels[indexes[math.floor(counter / 3)]]))
    plt.imshow(img, cmap='gray')
plt.show()

print('Sorting...')
sorted_images = sorted(x_test, key=lambda image: model.evaluate(image.reshape(1, 1, 28, 28), image.reshape(1, 1, 28, 28), verbose=0))
print('Sorted')

# Most standard data
columns = 10
rows = 12
images = [sorted_images[i].reshape((28, 28)) for i in range(columns*rows)]
fig = plt.figure(figsize=(28, 28))
for counter, i in enumerate(range(1, columns*rows + 1)):
    img = images[counter]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gray')
plt.show()

# Most abnormal data; anomalies
columns = 10
rows = 12
images = [sorted_images[-(i+1)].reshape((28, 28)) for i in range(columns*rows)]
fig = plt.figure(figsize=(28, 28))
for counter, i in enumerate(range(1, columns*rows + 1)):
    img = images[counter]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gray')
plt.show()
