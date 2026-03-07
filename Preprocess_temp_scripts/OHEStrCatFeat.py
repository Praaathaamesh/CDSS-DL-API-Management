# Given code is template for one-hot encoding of string categorical feature

import tensorflow as tf
import numpy as np

# scheme data
DictData = tf.constant([['a'],['b'],['c'],['a'],['b'],['c']])

# Deploy string lookup
encoder = tf.keras.layers.StringLookup(output_mode = "one_hot")
encoder.adapt(DictData)

# Begin testing
TestData = tf.constant([['a'],['r'],['c'],[' '],['b'],['c']])
EncodedData = encoder(TestData)

print(EncodedData)
