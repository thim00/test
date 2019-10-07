from __future__ import absolute_import, division, print_function, unicode_literals

import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import tfmodel


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


# Create directory for saving checkpoints.
checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        periods=5,
        verbose=1)

# Train the model using training data.
model.fit(train_images, train_labels, epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback])


# Test the model accuracy.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print()

