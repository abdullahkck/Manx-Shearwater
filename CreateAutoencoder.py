from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import numpy as np
import os
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pickle_in = open("Calls_67_pairs_X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Calls_67_pairs_y.pickle", "rb")
y = pickle.load(pickle_in)

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

X = np.array(X)
X_train = np.array(X_train)
X_test = np.array(X_test)

X = X/255.0
X_train = X_train/255.0
X_test = X_test/255.0

input_img = Input(shape=(640, 480, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)

x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[TensorBoard(log_dir='logs/full_autoencoder_e30')])

# -----------------------------------------------
# Get the encoder layer
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
# Get the encoded inputs for unsupervised learning
X_encoded_test = encoder.predict(X_test)  # Just to plot
X_encoded = encoder.predict(X)

# Save encoded inputs for unsupervised learning
pickle_out = open("Calls_67_pairs_X_encoded_30e_(4x40x30).pickle", "wb")
pickle.dump(X_encoded, pickle_out)
pickle_out.close()
# -----------------------------------------------

decoded_imgs = autoencoder.predict(X_test)

m = 6
n = 5
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(m, n, i + 1 + (n * 0))
    plt.imshow(X_test[i].reshape(480, 640))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded conv 1
    ax = plt.subplot(m, n, i + 1 + (n * 1))
    plt.imshow(X_encoded_test[i].reshape(30, 40, 4)[:, :, 0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded conv 2
    ax = plt.subplot(m, n, i + 1 + (n * 2))
    plt.imshow(X_encoded_test[i].reshape(30, 40, 4)[:, :, 1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded conv 3
    ax = plt.subplot(m, n, i + 1 + (n * 3))
    plt.imshow(X_encoded_test[i].reshape(30, 40, 4)[:, :, 2])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded conv 4
    ax = plt.subplot(m, n, i + 1 + (n * 4))
    plt.imshow(X_encoded_test[i].reshape(30, 40, 4)[:, :, 3])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(m, n, i + 1 + (n * 5))
    plt.imshow(decoded_imgs[i].reshape(480, 640))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
