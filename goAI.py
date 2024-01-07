import sys
import uuid
from pathlib import Path

from tensorflow import keras
import tensorflow as tf
from keras import layers
import random
import os
import numpy as np
import pickle


# Returns move x and y coordinates
def process_game_file(path):
    file = open(path).read()
    strings = file.split(';')
    moves = []
    for s in strings:
        if s.startswith(("B", "W")):
            try:
                x = (ord(s[2])) - 97
                y = (ord(s[3])) - 97
                moves.append((x, y))
            except IndexError:
                moves.append((19, 19))
    return moves


# Get data in flat form
def get_x_and_y_data_from_file(path):
    # print("processing file at path: " + path)
    game_moves = process_game_file(path)
    turn = 0
    if len(game_moves) == 0:
        # print("no moves found in file")
        return [], []

    def move_index(x, y):
        if not (x == 19 and y == 19):
            index = x * 19 + y
        else:
            index = 361
        return index

    # 362 nodes: 361 for squares, and 0 or 1 for whose turn
    # each board state is the tuple (input, output)
    states = [np.zeros(362)]
    moves = [move_index(game_moves[0][0], game_moves[0][1])]
    # encode board square as 0 for empty, 1 for black, -1 for white
    for move in game_moves[1:]:
        prev_state = states[len(states) - 1]
        new_state = prev_state.copy()
        flat_index = move[0] * 19 + move[1]
        # 19, 19 denotes pass
        if not (move[0] == 19 and move[1] == 19):
            if turn == 0:
                new_state[flat_index] = 1
            else:
                new_state[flat_index] = -1
        turn = 1 - turn
        new_state[361] = turn
        index = move_index(move[0], move[1])
        if index in range(0, 362):
            states.append(new_state)
            moves.append(index)
    return states, moves


class Stuff:

    # training: True if generating training dataset, false if generating test dataset
    def __init__(self, model, path, training=True):
        self.model = model
        self.path = path
        self.training = training


    fileName = 0
    write_to_file = False # If false, recursive_get_training_data will be a generator instead
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []

    def write_data_to_numpy(self, states, moves):
        for i, state in enumerate(states):
            yield state, moves[i]

    def write_data_to_file(self, states, moves):
        # print("states in file: " + str(len(states)))
        for i, state in enumerate(states):
            with open("/home/naglis/trainingData/data/" + str(self.fileName) + ".npy", "w") as f:
                f.write(str(state))

            with open("/home/naglis/trainingData/labels/" + str(self.fileName) + ".npy", "w") as f:
                f.write(str(moves[i]))

            self.fileName = self.fileName + 1

    # Write all training data from directory to file
    def recursive_get_training_data(self, path):
        if os.path.isfile(path):
            states, moves = get_x_and_y_data_from_file(path)
            yield from self.write_data_to_numpy(states, moves)
        if os.path.isdir(path):
            # x_data, y_data = [], []
            with os.scandir(path) as entries:
                for entry in entries:
                    yield from self.recursive_get_training_data(path=str(entry.path))
                    # x_data = x_data + data[0]
                    # y_data = y_data + data[1]
            # return x_data, y_data
        # return [], []

    def get_all_training_data(self):
        return self.recursive_get_training_data(self.path)

def model():
    model = keras.Sequential([
        # layers.Input(shape=(362,)),
        layers.Dense(362, activation="relu", name="input"),
        layers.Dense(350, activation="relu", name="foo"),
        layers.Dense(300, activation="relu"),
        layers.Dense(300, activation="relu"),
        layers.Dense(350, activation="relu"),
        layers.Dense(362, name="lastlayer")
    ])
    # y = model(x)
    # print(y[0][0:10])
    # print("number of weights:" + str(len(model.weights)))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train_model(datapath="goGames/games"):
    compiled_model = model()
    s = Stuff(model=compiled_model, path=datapath)
    ds_counter = tf.data.Dataset.from_generator(
        s.get_all_training_data,
        output_signature=(
            tf.TensorSpec(shape=(362,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32))
    )


    # for count_batch in ds_counter.repeat().batch(10).take(10):
    #     print(count_batch[0].numpy())
    #     print(np.shape(count_batch[0].numpy()))
    #     print(count_batch[1].numpy())

    s.model.fit(
        ds_counter.repeat(1000).batch(256),
        batch_size=512,
        epochs=1000,
        steps_per_epoch=60000
    )

    with open('model.pickle', 'wb') as handle:
        pickle.dump(compiled_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return compiled_model

def load_model():

    with open('model.pickle', 'rb') as handle:
        m = pickle.load(handle)
    return m

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_model()
