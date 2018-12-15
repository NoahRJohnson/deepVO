import argparse
import keras as K
import numpy as np
import os
import signal
import sys

from epoch import Epoch
from keras.layers import Dense, Activation, MaxPooling2D, Dropout, LSTM, Flatten, merge, TimeDistributed
from subseq_preds_to_full_pred import subseq_preds_to_full_pred
from time import time

from keras.layers import Concatenate

from keras.layers.convolutional import Conv2D

ap = argparse.ArgumentParser()

ap.add_argument('--batch_size', type=int, default=1)
ap.add_argument('--beta', type=int, default=100, help='Weight on orientation loss')
ap.add_argument('--data_dir', type=str, default='data/dataset', help='Where KITTI data is stored')
ap.add_argument('--hidden_dim', type=int, default=100, help='Dimension of LSTM hidden state')
ap.add_argument('--layer_num', type=int, default=2, help='How many LSTM layers to stack')
ap.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent optimization')
ap.add_argument('--num_epochs', type=int, default=5, help='How many full passes to make over the training data')
ap.add_argument('--step_size', type=int, default=10, help='How many optical flow samples to skip between subsequences.')
ap.add_argument('--subseq_length', type=int, default=50, help='How many optical flow images to include in one subsequence during training. Affects memory consumption.')
ap.add_argument('--mode', default = 'train', help="train or test. Train produces model checkpoints, test outputs csvs of poses for each testing sequence.")
ap.add_argument('--snapshot_dir', default='snapshots/', help='what folder to store model snapshots in')

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

        ### LOSS SO NEW IT WRAPS BACK AROUND TO BEING OLD ###
        
        squared_diff = K.backend.square(y_pred - y_true)

        y_shape = K.backend.int_shape(squared_diff)
        y_shape = (args['batch_size'],) + y_shape[1:]
        #print("y_shape")
        #print(y_shape)

        weights = np.ones(y_shape)
        weights[..., 3:] = beta
        weights = K.backend.variable(weights)

        # element-wise multiplication
        squared_diff_weighted = squared_diff * weights

        loss = K.backend.mean(squared_diff_weighted, axis=-1)
        

        ### NEWER LOSS ###
        """
        diff_abs = K.backend.abs(y_pred - y_true)

        y_shape = K.backend.int_shape(diff_abs)
        y_shape = (args['batch_size'],) + y_shape[1:]
        print("y_shape")
        print(y_shape)

        position_identity = np.zeros(y_shape)
        position_identity[..., 0:3] = 1
        position_identity = K.backend.variable(position_identity)

        position_diff_abs = diff_abs * position_identity

        orientation_identity = np.zeros(y_shape)
        orientation_identity[..., 4:7] = 1
        orientation_identity = K.backend.variable(position_identity)

        orientation_diff_abs = diff_abs * orientation_identity

        orientation_diff_abs_other_way = K.backend.variable(np.full(y_shape, 2*np.pi)) - orientation_diff_abs

        orientation_diff_abs_other_way = orientation_diff_abs_other_way * orientation_identity

        orientation_diff_magnitude = K.backend.minimum(orientation_diff_abs,
                                                       orientation_diff_abs_other_way)

        # Scale values by beta
        orientation_diff_magnitude = orientation_diff_magnitude * np.sqrt(beta)

        combined_diff = position_diff_abs + orientation_diff_magnitude

        loss = K.backend.mean(K.backend.square(combined_diff), axis=-1)
        """

        ###
        ## NEW LOSS WHICH HANDLES ANGLES
        ##
        """
        p_true = K.backend.gather(y_true, K.backend.variable(np.array([0,1,2]), dtype='int32'))
        p_pred = K.backend.gather(y_pred, K.backend.variable(np.array([0,1,2]), dtype='int32'))
        q_true = K.backend.gather(y_true, K.backend.variable(np.array([3,4,5]), dtype='int32'))
        q_pred = K.backend.gather(y_pred, K.backend.variable(np.array([3,4,5]), dtype='int32'))

        L_x = K.backend.mean(K.backend.square(p_pred - p_true), axis=-1)

        q_diff_abs = K.backend.abs(q_pred - q_true)

        q_diff_other_way = K.backend.variable(np.full(K.backend.int_shape(q_diff_abs), 2*np.pi)) - q_diff_abs

        q_magnitude = K.backend.minimum(q_diff_abs, q_diff_other_way)

        L_q = K.backend.mean(K.backend.square(q_magnitude), axis=-1)

        loss = L_x + beta * L_q
        """
        ####
        ###OLD LOSS
        ####
        """
        # Take the difference of each pose label and its estimate,
        # and square that element-wise
        squared_diff = K.backend.square(y_pred - y_true)

        # Multiply the orientations by beta, and sum
        # each tensor up
        weights = K.backend.variable(np.array([1,1,1,beta,beta,beta]))
        loss = K.backend.squeeze(K.backend.dot(squared_diff, K.backend.expand_dims(weights)), axis=-1)
        """
        return loss
    return weighted_mse 

# Separate the sequences for which there is ground truth into test 
# and train according to the paper's partition. 
train_seqs = ['00', '02', '08', '09'] 
test_seqs = ['03', '04', '05', '06', '07', '10']

# Create a data loader to get batches one epoch at a time
epoch_data_loader = Epoch(datadir=args['data_dir'],
                          flowdir=os.path.join(args['data_dir'], "flows"),
                          train_seq_nos=train_seqs,
                          test_seq_nos=test_seqs,
                          window_size=args['subseq_length'],
                          step_size=args['step_size'],
                          batch_size=args['batch_size'])

# What is the shape of the input flow images?
flow_input_shape = epoch_data_loader.get_input_shape()

# Define Keras model architecture
model = K.models.Sequential()

# Reducing input dimensions via conv-pool layers
model.add(TimeDistributed(Conv2D(10,(3,3)),
          input_shape=(args["subseq_length"], *flow_input_shape)))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(data_format="channels_first",
                                       pool_size=(7, 7))))

model.add(TimeDistributed(Conv2D(10,(3,3))))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(data_format="channels_first",
                                       pool_size=(5, 5))))

model.add(TimeDistributed(Conv2D(10,(3,3))))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(data_format="channels_first",
                                       pool_size=(5, 5))))

model.add(TimeDistributed(Conv2D(10,(3,3))))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(data_format="channels_first",
                                       pool_size=(3, 3))))

# Flatten outputs for input to LSTM
model.add(TimeDistributed(Flatten()))

# Stacked LSTM layers
for i in range(args['layer_num']):
    model.add(LSTM(args['hidden_dim'], return_sequences=True))

# A single dense layer to convert the LSTM output into
# a pose estimate vector of length 6. We use the default
# linear activation because pose position values can be
# unbounded.
model.add(TimeDistributed(Dense(6)))


# Compile the model, with custom loss function
model.compile(loss=custom_loss_with_beta(beta=args['beta']),
              optimizer=K.optimizers.Adam(lr=args['learning_rate']))
#model.compile(loss = "mse", optimizer = "adam")

# Describe layers and parameters
print("Model summary:")
print(model.summary())

# Set where weights and optimizer state are saved and loaded from
snapshot_path = os.path.join(args['snapshot_dir'],
			                 'model.h5')

# Load snapshot if it exists
if os.path.isfile(snapshot_path):
    print("Loading snapshot found at {}".format(snapshot_path))
    model = K.models.load_model(snapshot_path, custom_objects={'weighted_mse': custom_loss_with_beta(beta=args['beta'])})
else:
    # We can't test the network if we haven't already trained it
    if args['mode'] == 'test':
        print("ERROR: Trying to test network but snapshot file {} not found.".format(snapshot_path))
        sys.exit()

# Create TensorBoard
tensorboard = K.callbacks.TensorBoard(
    log_dir="logs/{}".format(time()))

# Attach it to our model
tensorboard.set_model(model)

# Create signal handler to catch Ctrl-C
# and save model before shutdown
def signal_handler(sig, frame):
        model.save(snapshot_path)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if args['mode'] == 'train':
    batch_num = 0
    for epoch in range(args['num_epochs']):

        losses = []

        while not epoch_data_loader.is_complete():

            # Get batch of random samples (subsequences)
            X, Y = epoch_data_loader.get_training_batch()

            # Update weights, and get training loss on this batch
            loss = model.train_on_batch(X, Y)

            # Store loss for epoch summary
            losses.append(loss)

            # Some console lovin
            print("[Epoch {} Batch {}] TRAINING LOSS: {}".format(epoch,
                                                                 batch_num,
                                                                 loss))

            # save loss history for this batch
            # on_batch_end doesn't work!? using on_epoch_end temporarily
            tensorboard.on_epoch_end(batch_num, dict(batch_training_loss=loss))
            batch_num += 1

        # Re partition and shuffle samples
        epoch_data_loader.reset()

        # Calculate average loss of all samples this epoch
        mean_epoch_loss = np.mean(losses)

        print("Epoch {} finished. AVG TRAINING LOSS: {}".format(epoch,
                                                                mean_epoch_loss))

        # save loss history with tensorboard at the end of each epoch
        #tensorboard.on_epoch_end(epoch, dict(epoch_training_loss=mean_epoch_loss))

    # Once we're done with training, save the model
    print("TRAINING FINISHED. SAVING SNAPSHOT TO {}".format(snapshot_path))
    model.save(snapshot_path)

    # And tell tensorboard
    tensorboard.on_train_end(None)

elif args['mode'] == 'test':

    for kitti_seq in test_seqs:

        # Open output file to write pose results to
        out_fname = "test_results/{}.csv".format(kitti_seq)

        losses = []
        estimated_poses = []
        for X, Y in epoch_data_loader.get_testing_samples(kitti_seq):

            # batch size of 1
            X = X[np.newaxis, :]

            #print("TESTING X SIZE = {}".format(X.shape))

            # get pose estimate
            estimated_batch = model.predict_on_batch(X)

            # TODO: Fix this to be more general for different batch sizes
            estimated_pose = estimated_batch[0]

            estimated_poses.append(estimated_pose)


            # Get testing loss
            Y = Y[np.newaxis, :]  # batch size of 1
            loss = model.test_on_batch(X, Y)
            #print("TESTING LOSS: {}".format(loss))
            losses.append(loss)

        # Calculate average loss of this sequence
        mean_seq_loss = np.mean(losses)

        print("Testing sequence {} finished. AVG TEST LOSS: {}".format(\
                                                                kitti_seq,
                                                                mean_seq_loss))

        # Put poses back in original reference frame
        # and write out to file
        subseq_preds_to_full_pred(estimated_poses, out_fname)

        # save loss history with tensorboard at the end of each sequence
        #tensorboard.on_epoch_end(epoch, dict(testing_loss=mean_loss))

else:
    print("ERROR: Mode {} not recognized".format(args['mode']))

