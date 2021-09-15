import pandas as pd
import pickle
import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import math


d=pd.DataFrame()


with open('/lustre/home/aierbad27/aiman/data_atlas.pkl', 'rb') as f:
  while True:   
    try:
        current_id=pickle.load(f)
        d = d.append(current_id, ignore_index=True)
    except EOFError:
        print('Pickle ends')
        break
		
		
print(d.count())
print(d[img][0].shape)

alls = d.category.value_counts().to_dict()
less_10 = []
for i,v in alls.items():
  if v<5:
    less_10.append(i)

d2 = d[~d['category'].isin(less_10)]

d2=pd.concat([d2,d2])
print(d2.count())

d2= d2.sample(frac=1)

categs = d2.category.unique()
num=[]
for i in range(0,len(categs)):
  num.append(i)

d2['category'].replace(categs,
                      num, inplace=True)

  

x_data_train=d2['img'][:17000].to_list()
x_data_test=d2['img'][17000:19000].to_list()
x_data_fd=d2['img'][19000:].to_list()

y_data_train=d2['category'][:17000].to_list()
y_data_test=d2['category'][17000:19000].to_list()
y_data_fd=d2['category'][19000:].to_list()

x_data_test_names=d2['img_name'][17000:19000].to_csv('x_data_test_names.csv')
x_data_fd_names=d2['img_name'][19000:].to_csv('x_data_fd_names.csv')



x_data_train = np.array(x_data_train)
y_data_train = np.array(y_data_train)
x_data_test = np.array(x_data_test)
y_data_test = np.array(y_data_test)
x_data_fd = np.array(x_data_fd)
y_data_fd = np.array(y_data_fd)


print(x_data_train.shape)
print(x_data_test.shape)
print(x_data_fd.shape)


base_model = keras.applications.ResNet50(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False,
)  

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))



x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.5)(x)  # Regularize with dropout
outputs = keras.layers.Dense(len(categs))(x)
model = keras.Model(inputs, outputs)

print(model.summary())


opt=keras.optimizers.Adam(0.0001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.run_eagerly = True



callbacks = [EarlyStopping(monitor='val_loss', patience=8), ModelCheckpoint(filepath='atlas_model.h5', monitor='val_loss', save_best_only=True)]

hist = model.fit(x_data_train, y_data_train, batch_size=None, epochs=50,
          validation_data=(x_data_test, y_data_test), 
          verbose=1, callbacks=callbacks)

