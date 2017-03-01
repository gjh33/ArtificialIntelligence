# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:57:51 2016

Machine Learning 551 Project 4
Model

This file is responsible for windowing then training a model

@author: Luke Jones
"""

import numpy
from sklearn import preprocessing
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
import mapper

#def convert_to_output(normalized):
#    regular = scaler.inverse_transform(numpy.array(normalized))
    

###CONFIGURATION###
random_seed = 7
# Path to data files
data_path = 'C:\\Users\\THEANO\\Documents\\Comp 551\\Project 4\\data\\'
# Window size of time data
window_size = 5
###END CONFIGURATION###

# Initialize RNG
numpy.random.seed(random_seed)

# Open files with data
training_file = open(data_path + 'train_features.csv')
test_file = open(data_path + 'test_features.csv')

# Load scaler used to encode data
with open('scaler.pkl') as pickle_file:
    scaler = pickle.load(pickle_file)

raw_train_data = numpy.genfromtxt(training_file, delimiter=',', dtype='str')
raw_test_data = numpy.genfromtxt(test_file, delimiter=',', dtype='str')

# group the data via id tag (so we dont window data from 2 different animals)
grouped_train = {}
for sample in raw_train_data:
    if sample[1] not in grouped_train:
        grouped_train[sample[1]] = []
    feature_set = map(float, sample[2:])
    grouped_train[sample[1]].append(feature_set)
grouped_test = {}
for sample in raw_train_data:
    if sample[1] not in grouped_test:
        grouped_test[sample[1]] = []
    feature_set = map(float, sample[2:])
    grouped_test[sample[1]].append(feature_set)

# window the data
train_x = []
train_y = []
test_x = []
test_y = []

# So we can keep window data grouped for predictions later
grouped_window_test = {}
for animal in grouped_test.keys():
    grouped_window_test[animal] = { "x": [], "y": [] }

for animal, data in grouped_train.items():
    for i in range(0, (len(data) - (window_size-1))):
        train_x.append(data[i:(i+window_size-1)])
        train_y.append(data[i+window_size-1][:2])

for animal, data in grouped_test.items():
    for i in range(0, (len(data) - (window_size-1))):
        test_x.append(data[i:(i+window_size-1)])
        test_y.append(data[i+window_size-1][:2])
        grouped_window_test[animal]["x"].append(data[i:(i+window_size-1)])
        grouped_window_test[animal]["y"].append(data[i+window_size-1][:2])
    grouped_window_test[animal]['y'] = scaler.inverse_transform(grouped_window_test[animal]['y'])
        
train_x = numpy.array(train_x)
train_y = numpy.array(scaler.inverse_transform(train_y))
test_x = numpy.array(test_x)
test_y = numpy.array(scaler.inverse_transform(test_y))

# Shuffle data
train = [[train_x[i], train_y[i]] for i in range(0, len(train_x))]
#test = [[test_x[i], test_y[i]] for i in range(0, len(test_x))]
numpy.random.shuffle(train)
#numpy.random.shuffle(test)
train_x = numpy.array([example[0] for example in train])
train_y = numpy.array([example[1] for example in train])


# Build the model
model = Sequential()
model.add(LSTM(1024, input_dim=len(train_x[0][0]), return_sequences=True,))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

filepath="serialized_models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(train_x, train_y, nb_epoch=100, batch_size=512, verbose=2, callbacks=callback_list, validation_split=0.33)

def evaluate_predictions(animal):
    cur_sample = grouped_window_test[animal]['x'][0]

    # Generating off of previous predictions. P(x) relies on P(x-1), P(x-2), P(x-3), ... etc
    predictions = []
    
    # Every prediction is made off of the next set of sample points. P(x) relies on sample(x)
    fake_predictions = []
    
    for i in range(0, len(grouped_window_test[animal]['y'])):
        res = model.predict(numpy.array([cur_sample]), batch_size=1)
        res_fake = model.predict(numpy.array([grouped_window_test[animal]['x'][i]]))
        predictions += list(res)
        fake_predictions += list(res_fake)
        new_sample = cur_sample[1:] + [list(scaler.transform(res)[0]) + list(cur_sample[0][2:])]
        cur_sample = new_sample
    
    combined = list(grouped_window_test[animal]['y']) + list(predictions) + list(fake_predictions)
    labels = list((['real'] * len(grouped_window_test[animal]['y'])) + (['generated'] * len(predictions)) + (['individually predicted'] * len(fake_predictions)))
    mapper.createMap(numpy.array(combined), numpy.array(labels))