from __future__ import print_function

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

tf.config.optimizer.set_jit(True)

print(np.__version__)
print(pd.__version__)
print(tf.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
print()



city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
pop_log = np.log(population)
print(type(pop_log))
print(pop_log)
print()

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(cities)
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Large city'] = cities['Area square miles'].apply(lambda val: val > 50.0)
print(cities)
print()

print(population > 1000000)
print()

#cities_desc = cities.describe()
#print(cities_desc)

#print(type(cities['City name']))
#print(cities['City name'])

#california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
#print(california_housing_dataframe.describe())


# creates nodes in a graph
# "construction phase"
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)
print(result)
# defines our session and launches graph
sess = tf.compat.v1.Session()
# runs result
print(sess.run(result))




# Simple learning model for matching linear function.
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
    ])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=100)
print(model.predict([10.0]))


# Simple learning model for matching boolean function.
model = tf.keras.Sequential([
    keras.layers.Dense(2, input_shape=[2], activation='relu'),
    keras.layers.Dense(1)
    ])
model.compile(optimizer='sgd', loss='mean_squared_error')

model2 = tf.keras.Sequential([
    keras.layers.Dense(2, input_shape=[2], activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model2.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

xs = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
ys = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
#inputs = np.array([np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0,0]), np.array([1.0, 1.0])])
#inputs = np.array([ [0.0, 0.0], [0.0, 1.0], [1.0, 0,0], [1.0, 1.0], ])
inputs = np.stack([ xs, ys ], axis=1)
zs = np.array([0, 0, 0, 1])

model.fit(inputs, zs, epochs=500)
model2.fit(inputs, zs, epochs=500)

print(inputs)
print(type(inputs))
print(inputs.shape)
print(type(inputs[0]))
print(inputs[0].shape)
print(inputs[0])

test_xs = np.array([0.0], dtype=float)
test_ys = np.array([0.0], dtype=float)
test_input = np.stack([test_xs, test_ys], axis=1)

print("Model 1 predictions: ", model.predict(inputs))
print("Model 2 predictions: ", model2.predict(inputs))


# Evaluate the model
#loss,acc = model.evaluate(inputs, zs, verbose=2)
#print("Model accuracy: {:5.2f}%".format(100*acc))
#print()


