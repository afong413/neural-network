#!/usr/bin/env python3

import numpy as np  # Not turtle
from tqdm import tqdm  # Progress bars
from neuralnet import Network, CalcFunction
from neuralnet.dense import DenseLayer
from sample_names import sample_names

# SETTING: In sample_names.py (not sample_names.txt), comment/uncomment
# the types of images you want/don't want.

# MARK: Get Samples

num_sample_types = len(sample_names)

samples = []

for name in tqdm(sample_names, desc="Reading data..."):
    try:
        samples.append(
            np.load(
                "samples//" + str(name.replace(" ", "_")) + ".npy",
            )
        )
    except FileNotFoundError:
        raise FileNotFoundError("Please run ./get_samples.sh.")

# MARK: Split Data

training_images = []
training_labels = []
testing_images = []
testing_labels = []

for i in tqdm(range(len(samples)), desc="Splitting data..."):
    np.random.shuffle(samples[i])
    testing_images += list(samples[i][: len(samples[i]) // 10] / 255)
    testing_labels += np.full(len(samples[i]) // 10, i).tolist()
    training_images += list(samples[i][len(samples[i]) // 10 :] / 255)
    training_labels += np.full(
        len(samples[i]) - (len(samples[i]) // 10), i
    ).tolist()

# MARK: Functions

sigmoid = CalcFunction(
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: (ex := np.exp(-x)) / ((1 + ex) ** 2),
)

xavier = lambda n_in, n_out: np.random.normal(
    0, np.sqrt(2 / (n_in + n_out)), (n_out, n_in)
)

squared_error = CalcFunction(
    np.vectorize(lambda v_out, expected_v_out: (v_out - expected_v_out) ** 2),
    np.vectorize(lambda v_out, expected_v_out: 2 * (v_out - expected_v_out)),
)

# MARK: Network

network = Network(
    [
        DenseLayer(784, 32, 0.05, 0.8, sigmoid, xavier),
        DenseLayer(32, num_sample_types, 0.05, 0.8, sigmoid, xavier),
    ],
    squared_error,
)

# MARK: Training

n_epochs = 10  # SETTING: How many times the model sees each sample.
epoch_size = len(training_images)
batch_size = 10000  # SETTING: The size of each batch.

shuffle = list(range(epoch_size))

for i in (bar := tqdm(range(n_epochs), desc="Training...")):
    np.random.shuffle(shuffle)
    l = 0

    for j in tqdm(
        range(epoch_size // batch_size), desc=f"Epoch {i + 1}...", leave=False
    ):
        correct = 0
        for k in tqdm(range(batch_size), leave=False):
            v_out = network(training_images[shuffle[batch_size * j + k]])

            expected_v_out = np.zeros(num_sample_types)
            expected_v_out[training_labels[shuffle[batch_size * j + k]]] = 1

            network.backprop(expected_v_out)

            l += 1

            if (
                np.argmax(v_out)
                == training_labels[shuffle[batch_size * j + k]]
            ):
                correct += 1

        bar.set_description(
            f"Training... (Acc: {round(100 * correct / batch_size, 2)}%)"
        )

        network.update(batch_size)

# MARK: Testing

test_size = 10000  # SETTING: How many tests to do.

test = np.random.choice(len(testing_images), test_size)

correct = 0
for i in tqdm(test, desc="Testing..."):
    v_out = network(testing_images[i])
    if np.argmax(v_out) == testing_labels[i]:
        correct += 1

print(f"Accuracy: {round(100 * correct / test_size, 2)}%")
