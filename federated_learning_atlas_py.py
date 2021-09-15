# -*- coding: utf-8 -*-


import pandas as pd
import pickle

x_data_test=test_df['img'].to_list()
x_data_fd=fd_df['img'].to_list()

y_data_test=test_df['category'].to_list()
y_data_fd=fd_df['category'].to_list()

import numpy as np

x_data_test = np.array(x_data_test)
y_data_test = np.array(y_data_test)
x_data_fd = np.array(x_data_fd)
y_data_fd = np.array(y_data_fd)


print(x_data_test.shape)
print(x_data_fd.shape)

from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
from keras.callbacks import LearningRateScheduler


loaded_model = load_model('/content/drive/MyDrive/Atlas-data/atlas_model.h5')


opt=keras.optimizers.Adam(0.0001)
loaded_model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

loaded_model.evaluate(x_data_test, y_data_test)

from datetime import datetime
start = datetime.now()
loaded_model.fit(x_data_test[:100], y_data_test[:100], epochs=1)
end = datetime.now()
time_taken = end - start
print('conversion Time: ',time_taken)

loaded_model.summary()

loaded_model.evaluate(x_data_fd, y_data_fd)

from datetime import datetime
start = datetime.now()
loaded_model.predict(x_data_fd[0].reshape(1,224,224,3)).argmax(axis=-1)
end = datetime.now()
time_taken = end - start
print('Time: ',time_taken)

!pip uninstall --yes tensorboard tb-nightly

!pip install tensorflow_federated_nightly
!pip install nest_asyncio Set up an Image Classification Model 
!pip install tb-nightly  # or tensorboard, but not both

import nest_asyncio
nest_asyncio.apply()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle
import pandas as pd

x_data_fed = []
y_data_fed = []
x_data_fed2 = []
y_data_fed2 = []
x_data_fed3 = []
y_data_fed3 = []
x_data_fed4 = []
y_data_fed4 = []
data = []
data2 = []
data3= []
data4 = []

counter = 0
for i in range(50):
  x_data_fed.append([i.reshape(-1,224,224,3)[0] for i in x_data_fd[counter:counter+2]])
  y_data_fed.append([i.reshape(-1,1)[0] for i in y_data_fd[counter:counter+2]])
  counter=counter+2
for i in range(50):
  data.append(tf.data.Dataset.from_tensors(collections.OrderedDict(x=x_data_fed[i]  , y=y_data_fed[i] )))

print(counter)

for i in range(50):
  x_data_fed2.append([i.reshape(-1,224,224,3)[0] for i in x_data_fd[counter:counter+2]])
  y_data_fed2.append([i.reshape(-1,1)[0] for i in y_data_fd[counter:counter+2]])
  counter=counter+2
for i in range(50):
  data2.append(tf.data.Dataset.from_tensors(collections.OrderedDict(x=x_data_fed2[i]  , y=y_data_fed2[i] )))

print(counter)

for i in range(50):
  x_data_fed3.append([i.reshape(-1,224,224,3)[0] for i in x_data_fd[counter:counter+2]])
  y_data_fed3.append([i.reshape(-1,1)[0] for i in y_data_fd[counter:counter+2]])
  counter=counter+2
for i in range(50):
  data3.append(tf.data.Dataset.from_tensors(collections.OrderedDict(x=x_data_fed3[i]  , y=y_data_fed3[i] )))

print(counter)

for i in range(50):
  x_data_fed4.append([i.reshape(-1,224,224,3)[0] for i in x_data_fd[counter:counter+2]])
  y_data_fed4.append([i.reshape(-1,1)[0] for i in y_data_fd[counter:counter+2]])
  counter=counter+2
for i in range(50):
  data4.append(tf.data.Dataset.from_tensors(collections.OrderedDict(x=x_data_fed4[i]  , y=y_data_fed4[i] )))

print(counter)

print(data[0].element_spec)
print(data2[0].element_spec)
print(data3[0].element_spec)
print(data4[0].element_spec)

print('Number of client datasets: {l}'.format(l=len(data)))
print('First dataset: {d}'.format(d=data[0]))

def create_tff_model():

  input_spec = data[0].element_spec
  keras_model_clone = tf.keras.models.clone_model(loaded_model)
  return tff.learning.from_keras_model(
      keras_model_clone,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam())

state = fed_avg.initialize()

# Load our pre-trained Keras model weights into the global model state.
state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in loaded_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in loaded_model.non_trainable_weights
    ])

loaded_model2 = load_model('/content/drive/MyDrive/Atlas-data/atlas_model.h5', compile=False)

from datetime import datetime

def keras_evaluate(state, round_num):
  # Take our global model weights and push them back into a Keras model to
  # use its standard `.evaluate()` method.
  keras_model = loaded_model2
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  state.model.assign_weights_to(keras_model)
  loss, accuracy = keras_model.evaluate(x_data_fd[500:],y_data_fd[500:] )
  print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))

#from RegscorePy import *

print('Round {r}'.format(r=1))
start1 = datetime.now()
state, metrics = fed_avg.next(state, data)
time_taken = datetime.now() - start1
print('fd Time: ',time_taken) 
train_metrics = metrics['train']
print(" train mterics: ", train_metrics)
keras_evaluate(state, 1)


print('Round {r}'.format(r=2))
start1 = datetime.now()
state, metrics = fed_avg.next(state, data2)
time_taken = datetime.now() - start1
print('fd Time: ',time_taken) 
train_metrics = metrics['train']
print("mterics: ", train_metrics)
keras_evaluate(state, 2)


print('Round {r}'.format(r=3))
start1 = datetime.now()
state, metrics = fed_avg.next(state, data3)
time_taken = datetime.now() - start1
print('fd Time: ',time_taken) 
train_metrics = metrics['train']
print("mterics: ", train_metrics)
keras_evaluate(state, 3)


print('Round {r}'.format(r=4))
start1 = datetime.now()
state, metrics = fed_avg.next(state, data4)
time_taken = datetime.now() - start1
print('fd Time: ',time_taken) 
train_metrics = metrics['train']
print("mterics: ", train_metrics)
keras_evaluate(state, 4)
