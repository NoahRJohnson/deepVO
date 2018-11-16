import argparse
import keras as K
import numpy as np

from keras.callbacks import TensorBoard
from time import time

ap = argparse.ArgumentParser()

ap.add_argument('--batch_size', type=int, default=50)
ap.add_argument('--layer_num', type=int, default=2)
ap.add_argument('--seq_length', type=int, default=50)
ap.add_argument('--hidden_dim', type=int, default=500)
ap.add_argument('--generate_length', type=int, default=500)
ap.add_argument('--nb_epoch', type=int, default=20)
ap.add_argument('--mode', default = 'train')
ap.add_argument('--weights', default='')

args = vars(ap.parse_args())

def weighted_mse(y_true, y_pred):
        '''
        Custom loss function
        Return L_x + b*L_q
        '''
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

# Create TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(
