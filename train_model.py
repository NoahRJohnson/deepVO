import keras as K
import numpy as np


def weighted_mse(y_true, y_pred):
        '''Return L_x+bL_q.'''
        wy_pred = K.dot()([y_pred, K.variable(np.array([1,1,1, b, b, b]])
        wy_true = K.dot()([y_true, K.variable(np.array([1,1,1, b, b, b]])
        return K.losses.mean_squared_error(wy_true, wy_pred)

model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """1"""
                              filters=64,
                              activation='relu',
                              kernel_size=(7, 7 , 2),
                              data_format='channels_last')
                              padding='same',
                              strides=(2, 2, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """2"""
                              filters=128,
                              activation='relu',
                              kernel_size=(5, 5, 2),
                              data_format='channels_last')
                              padding='same',
                              strides=(2, 2, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """3"""
                             filters=256,
                             activation='relu',
                             kernel_size=(5, 5, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(2, 2, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """4"""
                             filters=256,
                             activation='relu',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(1, 1, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """5"""
                             filters=512,
                             activation='relu',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(2, 2, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """6"""
                             filters=512,
                             activation='relu',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(1, 1, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """7"""
                             filters=512,
                             activation='relu',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(2, 2, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """8"""
                             filters=512,
                             activation='relu',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(1, 1, 1))
model.add(K.layers.Conv3d(input_shape=(1392, 512, 2, 3), """9"""
                             filters=1024,
                             activation='none',
                             kernel_size=(3, 3, 2),
                             data_format='channels_last')
                             padding='same',
                             strides=(2, 2, 1))
model.add(K.layers.LSTM(6, return_sequences=True))
model.compile(loss=weighted_mse, optimizer='adam', metrics=['accuracy'])
