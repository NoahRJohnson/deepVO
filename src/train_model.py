import argparse
import batcher 
import keras as K
import numpy as np
import os
from epoch import Epoch
from time import time

ap = argparse.ArgumentParser()

ap.add_argument('data_dir', type=str, help='Where KITTI data is stored')
ap.add_argument('--batch_size', type=int, default=50)
ap.add_argument('--beta', type=int, default=100, help='Weight on orientation loss')
ap.add_argument('--hidden_dim', type=int, default=1000, help='Dimension of LSTM hidden state')
ap.add_argument('--layer_num', type=int, default=2, help='How many LSTM layers to stack')
ap.add_argument('--num_epochs', type=int, default=20, help='How many full passes to make over the training data')
ap.add_argument('--step_size', type=int, default=1, help='How many optical flow samples to skip between subsequences.')
ap.add_argument('--subseq_length', type=int, default=50, help='How many optical flow images to include in one subsequence during training. Affects memory consumption.')
ap.add_argument('--mode', default = 'train', help="train or test. Train produces model checkpoints, test outputs csvs of poses for each testing sequence.")
#ap.add_argument('--weights', default='')

args = vars(ap.parse_args())


def custom_loss_with_beta(beta):
    def weighted_mse(y_true, y_pred):
        """Custom loss function for jointly learning
        position and orientation.

        Args:
            y_true: The pose label
            y_pred: The estimated pose

        Returns:
            L_x + beta*L_q

            Where L_x is the position loss,
            L_q is the orientation loss,
            and beta is a hyperparameter
        """
        p_true = K.backend.gather(y_true, (0,1,2))
        p_pred = K.backend.gather(y_pred, (0,1,2))
        q_true = K.backend.gather(y_true, (3,4,5))
        q_pred = K.backend.gather(y_pred, (3,4,5))
        #L_x = K.losses.mean_squared_error(p_true, p_pred)
        #L_q = K.losses.mean_squared_error(q_true, q_pred)
        L_x = K.backend.mean(K.backend.square(p_pred - p_true), axis=-1)
        L_q = K.backend.mean(K.backend.square(q_pred - q_true), axis=-1)

        #wy_pred = K.dot(y_pred, K.variable(np.array([1,1,1, b, b, b])))
        #wy_true = K.dot(y_true, K.variable(np.array([1,1,1, b, b, b])))

        return (L_x + beta * L_q)
    return weighted_mse 

# Separate the sequences for which there is ground truth into test 
# and train according to the paper's partition. 
#train_seqs = ['00', '02', '08', '09'] 
train_seqs = ['00', '01', '02'] # until we finish generating
test_seqs = ['03', '04', '05', '06', '07', '10']

# Create a data loader to get batches one epoch at a time
epoch_data_loader = Epoch(datadir=args['data_dir'],
                          flowdir=os.path.join(args['data_dir'], "flows"),
                          train_seq_nos=train_seqs,
                          #test_seq_nos=test_seqs, #epoch doesn't take this as an argument in epoch.py
                          window_size=args['subseq_length'],
                          step_size=args['step_size'],
                          batch_size=args['batch_size'])

# How long are the flattened flow images?
num_features = epoch_data_loader.get_num_features()

# Define Keras model architecture
model = K.models.Sequential()

# Stacked LSTM layers
for i in range(args['layer_num']):
    model.add(K.layers.LSTM(args['hidden_dim'],
                            batch_input_shape=(args['batch_size'],
                                               args['subseq_length'],
                                              int( num_features)),
                            return_sequences=True))

# A single dense layer to convert the LSTM output into
# a pose estimate vector of length 6. We use a linear
# activation because pose position values can be
# unbounded.
model.add(K.layers.TimeDistributed(K.layers.Dense(6, activation='linear')))

# Compile the model, with custom loss function
model.compile(loss=custom_loss_with_beta(beta=args['beta']), optimizer='adam')

# #print-debugging lyfe
print("Layers: {}".format(model.layers))

# Create TensorBoard
tensorboard = K.callbacks.TensorBoard(
    log_dir="logs/{}".format(time()))

# Attach it to our model
tensorboard.set_model(model)

if args['mode'] == 'train':
    for epoch in range(args['num_epochs']):

        losses = []

        while not epoch_data_loader.is_complete():
            X, Y = epoch_data_loader.get_batch()
            loss = model.train_on_batch(x_i, y_i)  # update weights
            losses.append(loss)
            print("TRAINING LOSS: {}".format(np.mean(losses)))

        # Re partition and shuffle
        epoch_data_loader.reset()

        # Calculate average loss of all samples this epoch
        mean_epoch_loss = np.mean(losses)

        # save loss history with tensorboard at the end of each epoch
        tensorboard.on_epoch_end(epoch, dict(training_loss=mean_epoch_loss))

    tensorboard.on_train_end(None)

elif args['mode'] == 'test':
#  TODO
    for kitti_seq in test_seqs:

        # Open output file to write pose results to
        out_f = open('test_results/{}.csv'.format(kitti_seq))

        test_data_loader = Epoch()

        losses = []
        while not epoch_data_loader.is_complete():
            X, Y = epoch_data_loader.get_batch()

            loss = model.train_on_batch(x_i, y_i)  # update weights
            losses.append(loss)

            print("TRAINING LOSS: {}".format(np.mean(losses)))

        # Re partition and shuffle
        epoch_data_loader.reset()

        # Load subsequences to train on
        X, Y = batcher.get_samples(basedir=args['data_dir'],
                                           seq=kitti_seq,
                                           batch_size=1)
        for i in range(len(X)):  # looping over samples
            y_i = np.array([Y[i]])

            x_i = np.expand_dims(np.expand_dims(X[i], axis=1), axis=1)

            estimated_pose = predict_on_batch(x_i)  # get pose

            # write out pose to file
            out_f.write("{},{}".format(i, estimated_pose))

            model.test_on_batch(x_i, y_i)  # get testing loss
            losses.append(loss)


        # Clean up I/O
        out_f.close()

    # Calculate average loss of all samples in the testing data
    mean_loss = np.mean(losses)

    # save loss history with tensorboard at the end of each epoch
    tensorboard.on_epoch_end(epoch, dict(testing_loss=mean_loss))

else:
    print("ERROR: Mode {} not recognized".format(args['mode']))

