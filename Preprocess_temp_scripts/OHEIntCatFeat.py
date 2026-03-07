# Given is the code for OHE of integer discrete features

import tensorflow as tf

DictData = tf.constant([[10],[20],[30],[0],[10],[20]])

Encoder = tf.keras.layers.IntegerLookup()
Encoder.adapt(DictData)

TestData = tf.constant([[30],[0],[20],[10],[10],[50]])
EncodedData = Encoder(TestData)
print(EncodedData)

