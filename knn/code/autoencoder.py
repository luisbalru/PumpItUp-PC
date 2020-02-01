from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, UpSampling1D
from keras import regularizers

import numpy as np
import pandas as pd

import random
import tensorflow as tf

import os
os.environ['PYTHONHASHSEED']=str(123456789)

random.seed(123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

def fitTransform(X_train, X_test, nepochs, hid, bsize=32, drop_rate=0.2, kreg=regularizers.l2(0.01), areg=regularizers.l1(0.01), optimizer="adam", loss="mse"):
    input_img = Input(shape=(len(X_train[0]),))
    encoded = Dense(hid[0], activation='relu',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(input_img)
    encoded = Dropout(drop_rate)(encoded)
    for l in hid[1:]:
        encoded = Dense(l, activation='relu',
                    kernel_regularizer=kreg,
                    activity_regularizer=areg)(encoded)
        encoded = Dropout(drop_rate)(encoded)

    ilayers = hid[::-1][1:]
    decoded = Dense(ilayers[0], activation='relu',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(encoded)
    decoded = Dropout(drop_rate)(decoded)

    for l in ilayers[1:]:
        decoded = Dense(l, activation='relu',
                    kernel_regularizer=kreg,
                    activity_regularizer=areg)(decoded)
        decoded = Dropout(drop_rate)(decoded)

    decoded = Dense(len(X_train[0]), activation='sigmoid',
                kernel_regularizer=kreg,
                activity_regularizer=areg)(decoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    encoder.compile(optimizer=optimizer, loss=loss)

    autoencoder.fit(X_train, X_train,
                    epochs=nepochs,
                    batch_size=bsize,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1)
    X_train_reduced = encoder.predict(X_train)
    X_test_reduced = encoder.predict(X_test)
    return X_train_reduced, X_test_reduced
