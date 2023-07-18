import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Lambda, MaxPooling2D, GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model


class Models(object):

    @staticmethod    
    def dqn(inputs, action_space, lr):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(6, 6), strides=(1, 1), activation='relu', input_shape=inputs))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(action_space, activation='linear'))

        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam)

        return model