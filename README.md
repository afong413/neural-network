# neural-network

## About

This is an implementation of a neural network without using any pre-built librarys such as TensorFlow or Keras. Instead I built my own package (`neuralnet`) which handles the neural network propagation and backpropagation. This was a fun exercise in coding, linear algebra, and calculus and is decently fast as well.

## Usage

Install dependencies from `requirements.txt`.

### Library

Place the `neuralnet` directory in the same directory as your project.

### Example Implemenation

First, run `get_samples.sh` to fetch the sample files from Google Storage. This may take a while so feel free to remove unneeded samples from `sample_names.txt`. You can check `sample_names.py` for the categories that the model is currently trained to use. Then run `main.py`.
