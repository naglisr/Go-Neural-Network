import sys
import uuid
from pathlib import Path

from tensorflow import keras
import tensorflow as tf
from keras import layers
import random
import os
import numpy as np


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
    print("processing file at path: " + path)
    game_moves = process_game_file(path)
    turn = 0
    if len(game_moves) == 0:
        print("no moves found in file")
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
        states.append(new_state)
        moves.append(move_index(move[0], move[1]))
    return states, moves


class Stuff:
    fileName = 0
    write_to_file = False # If false, return numpy array in memory instead
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
    def write_data_to_dataset(self, states, moves):
        num = random.random()
        if num > 0.2:
            self.train_examples = self.train_examples + states
            self.train_labels = self.train_labels + moves
        else:
            self.test_examples = self.test_examples + states
            self.test_labels = self.test_labels + moves

    def write_data_to_file(self, states, moves):
        print("states in file: " + str(len(states)))
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
            if self.write_to_file:
                self.write_data_to_file(states, moves)
            else:
                self.write_data_to_dataset(states, moves)
        if os.path.isdir(path):
            # x_data, y_data = [], []
            with os.scandir(path) as entries:
                for entry in entries:
                    self.recursive_get_training_data(str(entry.path))
                    # x_data = x_data + data[0]
                    # y_data = y_data + data[1]
            # return x_data, y_data
        # return [], []


def model():
    model = keras.Sequential([
        # layers.Input((362,), name="input"),
        layers.Dense(362, activation="relu", name="input"),
        layers.Dense(350, activation="relu", name="foo"),
        layers.Dense(300, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(300, activation="relu"),
        layers.Dense(350, activation="relu"),
        layers.Dense(362, name="lastlayer")
    ])
    # y = model(x)
    # print(y[0][0:10])
    # print("number of weights:" + str(len(model.weights)))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train_model(datapath="goGames/games"):
    s = Stuff()
    # x_data, y_data = recursive_get_training_data("goGames/games")
    s.recursive_get_training_data(datapath)
    ds = tf.data.TFRecordDataset

    x_train, y_train = np.array(s.train_examples), np.array(s.train_labels)
    x_test, y_test = np.array(s.test_examples), np.array(s.test_labels)

    data_size = len(x_train)
    compiled_model = model()
    print(np.shape(x_train))
    print(np.shape(y_train))
    compiled_model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=2,
        validation_data=(x_train[-int(0.1*data_size):], y_train[-int(0.1*data_size):]))
    result = compiled_model.evaluate(x_test, y_test)

    print(dict(zip(compiled_model.metrics_names, result)))

    return compiled_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_model()
