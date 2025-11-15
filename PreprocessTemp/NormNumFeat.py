# Given is the template code for Normalising Numerical Inputs

import tensorflow as tf
import numpy as np

# Load the data
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape((len(x_train), -1))
input_shape = x_train.shape[1:]
classes = 10

# Scheme/set internal state of normalisation layer
Normaliser = tf.keras.layers.Normalization()
Normaliser.adapt(x_train)

# Instantiate the model
Input = tf.keras.Input(shape = input_shape)
Middle = Normaliser(Input)
Output = tf.keras.layers.Dense(classes, activation = "softmax")(Middle)

# Create/train the model
Model = tf.keras.Model(Input, Output)
Model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
Model.fit(x_train, y_train)
