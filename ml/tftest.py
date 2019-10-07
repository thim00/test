from __future__ import absolute_import, division, print_function, unicode_literals

import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import tfmodel

print(tf.__version__)
print(tf.version.VERSION)
print()

# Load fashion data.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Label data class names.
# Datasets are label as numbers which then translates into the following data class (category) names.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(train_labels.shape)
print()

# Preprocess the data to 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0


# Create a model.
model = tfmodel.createModel()
print(model.summary())
print()


# Test the model accuracy.
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model accuracy: {:5.2f}%".format(100*acc))
print()


# Create directory for retrieving checkpoints.
checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model accuracy: {:5.2f}%".format(100*acc))
print()

