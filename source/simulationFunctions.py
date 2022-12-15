#  -- a priori input sweep I_A
import pickle
import time

import numpy as np
from datetime import datetime
import pandas as pd
import tensorflow as tf

########################################
## This file contains utility functions
########################################


def save_data(sim_title, plot_data, sim_params=None, path="./fig/data/simulationResults/"):
    try:
        filename = datetime.now().strftime("%Y-%m-%d %H-%M ") + sim_title.replace("&", "").replace(".", "").replace(" ",
                                                                                                                    "")
        file = open(path + filename + ".csv", "w")
        df = pd.DataFrame.from_dict(plot_data)
        df.to_csv(file, line_terminator='\n')
        file.close()

        with open(path + filename + ".pickle", 'wb') as f:
            pickle.dump(plot_data, f)

        if sim_params is not None:
            file = open(path + filename + '_params.csv', "w")
            df = pd.DataFrame.from_dict(sim_params)
            df.to_csv(file, line_terminator='\n')
            file.close()

    except Exception as e:
        print(e)

def load_data(filename, path="./fig/data/simulationResults/"):
    with open(path + filename, "rb") as f:
        data = pickle.load(f)
    return data

def genRandIdx(numChan, numUes,batchSize):
    idx = np.arange(numChan, dtype=np.int)
    idx_ar = np.zeros((batchSize, numUes), dtype=np.int)
    for i in range(batchSize):
        idx_ar[i, :] = np.random.choice(idx, numUes, replace=False)
    return idx_ar

# Utility function for saving model weights
def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)

# Utility function for loading model weights
def load_weights(model, model_weights_path):
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

def train_model(model, ebno_db_min, ebno_db_max, num_training_iterations, training_batch_size):
    # Optimizer Adam used to apply gradients

    # optimizer = tf.keras.optimizers.SGD(momentum=0.2, learning_rate=0.05)
    optimizer = tf.keras.optimizers.Adam()
    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            bce = model(training_batch_size, ebno_db)
            loss_value = bce
        # Computing and applying gradients
        weights = model.trainable_weights
        # print(weights)
        grads = tape.gradient(loss_value, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 5 == 0:
            print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations, bce.numpy()))
            for s in zip(weights, grads):
                # print(np.mean(np.abs(s[0])), np.mean(np.abs(s[1])))
                print(s[0], s[1])
            # print("Weight: %.4f Gradient: %.4f" % (weights[1].numpy(), grads[1].numpy()))

def train_model_deweighting_SNR(model, snr_dB_min, snr_dB_max, training_batch_size, num_training_epochs, num_iter_per_epoch):
    # Optimizer Adam used to apply gradients

    # optimizer = tf.keras.optimizers.SGD(momentum=0.2, learning_rate=0.05)
    optimizer = tf.keras.optimizers.Adam()
    deweighting_weights = tf.Variable(tf.ones((training_batch_size))/training_batch_size, trainable=False, dtype=tf.float32, name="deweighting weights")
    ebno_db = tf.cast(tf.linspace(snr_dB_min, snr_dB_max, training_batch_size), dtype=tf.float32)
    for i_e in range(num_training_epochs):
        sum_loss=0
        print("Epoch {}/{}  Weights: \n".format(i_e, num_training_epochs) + str(deweighting_weights.numpy().transpose() ))
        for i_iter in range(num_iter_per_epoch):
            # Sampling a batch of SNRs
            # Forward pass
            with tf.GradientTape() as tape:
                bce = model(training_batch_size, ebno_db)
                sum_loss = sum_loss + bce
                loss_value = tf.reduce_sum(bce * deweighting_weights)
            # Computing and applying gradients
            weights = model.trainable_weights
            # print(weights)
            grads = tape.gradient(loss_value, weights)
            optimizer.apply_gradients(zip(grads, weights))
            # sum_bce = sum_bce + bce
            # Periodically printing the progress
            if i_iter % 5 == 0:
                print('Iteration {}/{}  BCE: {:.4f}'.format(i_iter, num_iter_per_epoch, loss_value.numpy()))
                for s in zip(weights, grads):
                    # print(np.mean(np.abs(s[0])), np.mean(np.abs(s[1])))
                    print(s[0], s[1])
                # print("Weight: %.4f Gradient: %.4f" % (weights[1].numpy(), grads[1].numpy()))
        deweighting_weights.assign(1/(sum_loss + 1e-4))
        deweighting_weights.assign(deweighting_weights/tf.reduce_sum(deweighting_weights))
        print(sum_loss)

def train_model_BLER(model_BCE, model_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations, num_BLER_training_iterations, training_batch_size):
    # Optimizer Adam used to apply gradients
    train_model(model_BCE, ebno_db_min, ebno_db_max, num_pretraining_iterations, training_batch_size)
    model_BLER.set_weights(model_BCE.get_weights())
    train_model(model_BLER, ebno_db_min, ebno_db_max, num_BLER_training_iterations, training_batch_size)

def train_model_BLER_SNR_deweighting(model_BCE, model_BLER, ebno_db_min, ebno_db_max, num_pretraining_iterations, num_snr_training_epochs, num_iter_per_epoch, training_batch_size):
    train_model(model_BCE, ebno_db_min, ebno_db_max, num_pretraining_iterations, training_batch_size)
    model_BLER.set_weights(model_BCE.get_weights())
    train_model_deweighting_SNR(model_BLER, ebno_db_min, ebno_db_max, training_batch_size, num_snr_training_epochs, num_iter_per_epoch)