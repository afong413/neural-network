#!/usr/bin/env python3


from pathlib import Path

import numpy as np  # Not turtle
from tqdm import tqdm  # Progress bars

from neuralnet import CalcFunction, Network
from neuralnet.dense import DenseLayer
from sample_names import sample_names

# SETTING: In sample_names.py (not sample_names.txt), comment/uncomment
# the types of images you want/don't want.

IMAGE_SIZE = 28 * 28

# MARK: Get Samples

num_sample_types = len(sample_names)

samples = []

for name in tqdm(sample_names, desc='Reading data...'):
    try:
        samples.append(np.load(Path('samples', name.replace(' ', '_') + '.npy')))
    except FileNotFoundError:
        raise FileNotFoundError('Please run ./get_samples.sh.')

# MARK: Split Data

training_images_parts, training_labels_parts = [], []
testing_images_parts, testing_labels_parts = [], []

for i in tqdm(range(len(samples)), desc='Splitting data...'):
    np.random.shuffle(samples[i])
    split = len(samples[i]) // 10
    testing_images_parts.append(samples[i][:split] / 255)
    testing_labels_parts.append(np.full(split, i))
    training_images_parts.append(samples[i][split:] / 255)
    training_labels_parts.append(np.full(len(samples[i]) - split, i))

training_images = np.vstack(training_images_parts)
training_labels = np.concatenate(training_labels_parts)
testing_images = np.vstack(testing_images_parts)
testing_labels = np.concatenate(testing_labels_parts)

# MARK: Functions

sigmoid = CalcFunction(
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: (ex := np.exp(-x)) / ((1 + ex) ** 2),
)

linear = CalcFunction(
    lambda x: x,
    lambda x: np.ones_like(x),
)


def xavier(n_in: int, n_out: int):
    return np.random.normal(0, np.sqrt(2 / (n_in + n_out)), (n_out, n_in))


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for numerical stability
    return e / e.sum(axis=-1, keepdims=True)


softmax_cross_entropy = CalcFunction(
    lambda v_out, expected_v_out: -np.sum(expected_v_out * np.log(softmax(v_out) + 1e-9)),
    lambda v_out, expected_v_out: softmax(v_out) - expected_v_out,
)

# MARK: Network

network = Network(
    [
        DenseLayer(IMAGE_SIZE, 32, 0.05, 0.8, sigmoid, xavier),
        DenseLayer(32, num_sample_types, 0.05, 0.8, linear, xavier),
    ],
    softmax_cross_entropy,
)

# MARK: Training

n_epochs = 10  # SETTING: How many times the model sees each sample.
epoch_size = len(training_images)
batch_size = 10000  # SETTING: The size of each batch.

shuffle = np.arange(epoch_size)

for i in (bar := tqdm(range(n_epochs), desc='Training...')):
    np.random.shuffle(shuffle)

    for j in tqdm(range(epoch_size // batch_size), desc=f'Epoch {i + 1}...', leave=False):
        batch = shuffle[batch_size * j : batch_size * (j + 1)]
        images = training_images[batch]
        labels = training_labels[batch]

        expected_v_out = np.zeros((batch_size, num_sample_types))
        expected_v_out[np.arange(batch_size), labels] = 1

        v_out = network(images)
        network.backprop(v_out, expected_v_out)

        correct = np.sum(np.argmax(v_out, axis=1) == labels)

        bar.set_description(f'Training... (Acc: {round(100 * correct / batch_size, 2)}%)')

        network.update(batch_size)

# MARK: Testing

test_size = 10000  # SETTING: How many tests to do.

test = np.random.choice(len(testing_images), test_size)

correct = 0
for i in tqdm(test, desc='Testing...'):
    v_out = network(testing_images[i])
    if np.argmax(v_out) == testing_labels[i]:
        correct += 1

print(f'Accuracy: {round(100 * correct / test_size, 2)}%')

Path('save').mkdir(exist_ok=True)
network.save('save/network.npz')
print('Saved to "save/network.npz"')
