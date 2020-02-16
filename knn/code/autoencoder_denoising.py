from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, UpSampling1D
from keras import regularizers

import numpy as np
import pandas as pd

import random
import tensorflow as tf
from keras.callbacks import EarlyStopping

import os
os.environ['PYTHONHASHSEED']=str(123456789)

random.seed(123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

def fitTransform(X_train, X_test, nepochs, hid, bsize=32, drop_rate=0.25, kreg=regularizers.l2(0.01), areg=regularizers.l1(0.01), optimizer="adam", loss="mse"):
    input_img = Input(shape=(len(X_train[0]),))
    encoded = Dropout(drop_rate)(input_img)
    encoded = Dense(hid[0], activation='tanh',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(encoded)
    for l in hid[1:]:
        encoded = Dense(l, activation='tanh',
                    kernel_regularizer=kreg,
                    activity_regularizer=areg)(encoded)

    ilayers = hid[::-1][1:]
    decoded = Dense(ilayers[0], activation='tanh',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(encoded)

    for l in ilayers[1:]:
        decoded = Dense(l, activation='tanh',
                    kernel_regularizer=kreg,
                    activity_regularizer=areg)(decoded)

    decoded = Dense(len(X_train[0]), activation='softmax',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(decoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    #callbacks = [EarlyStopping(monitor="val_loss", patience=3)]
    autoencoder.compile(optimizer=optimizer, loss=loss)
    encoder.compile(optimizer=optimizer, loss=loss)

    autoencoder.fit(X_train, X_train,
                    epochs=nepochs,
                    batch_size=bsize,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1)
    X_train_reduced = autoencoder.predict(X_train)
    X_test_reduced = autoencoder.predict(X_test)
    return X_train_reduced, X_test_reduced
