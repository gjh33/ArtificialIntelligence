# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:28:10 2016

Machine Learning 551 Project 4
Data formatter

This file is responsible for formattting and combining csv files into usable
data for machine learning

@author: Luke Jones
"""
import csv
import numpy
import dateutil.parser
from sklearn import preprocessing
import pickle

#####CONFIGURATION#####
# Files to combine
file_names = ['bald_eagle_1.csv', 'bald_eagle_2.csv', 'buzz_eagle.csv']
# Path to folder containing these files. Don't forget to escape backslashes.
raw_data_path = 'C:\\Users\\THEANO\\Documents\\Comp 551\\Project 4\\raw\\'
# Path to output folder for combined file
data_output_path = 'C:\\Users\\THEANO\\Documents\\Comp 551\\Project 4\\data\\'
# Columns to load. Make sure all files have these columns
column_names = ['event-id', 'location-long', 'location-lat', \
                'individual-taxon-canonical-name', 'tag-local-identifier', \
                'timestamp']
#####END CONFIGURATION#####

# Build combined output
output = []

for i in range(0, len(file_names)):
    with open(raw_data_path + file_names[i]) as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            output_row = {}
            valid = True
            for cname in column_names:
                if row[cname] == '':
                    valid = False
                output_row[cname] = row[cname]
            if valid:
                output.append(output_row)
output = sorted(output, key=lambda k: int(k[column_names[0]]))

# Convert into feature mapping
samples = []
for sample in output:
    feature_list = []
    feature_list.append(int(sample['event-id']))
    feature_list.append(int(sample['tag-local-identifier']))
    feature_list.append(float(sample['location-long']))
    feature_list.append(float(sample['location-lat']))
    feature_list.append(sample['individual-taxon-canonical-name'])
    feature_list.append(sample['timestamp'])
    samples.append(feature_list)

samples = numpy.array(samples)
scaler = preprocessing.MinMaxScaler()
location_norm = scaler.fit_transform(samples[:, 2:4].tolist())
with open('scaler.pkl', 'w+') as pickle_file:
    pickle.dump(scaler, pickle_file)
canon_name_col = samples[:, [4]].flatten()
canon_labels = numpy.unique(canon_name_col)
canon_labels = canon_labels.tolist()
canon_encoded = []
for label in canon_name_col:
    encoding = numpy.zeros(len(canon_labels))
    encoding[canon_labels.index(label)] = 1
    canon_encoded.append(encoding)
    
timestamps = samples[:, [5]].flatten()
def extract_month(timestamp):
    return dateutil.parser.parse(timestamp).month
months = map(extract_month, timestamps.tolist())
month_labels = numpy.unique(numpy.array(months)).tolist()
months_encoded = []
for label in months:
    encoding = numpy.zeros(len(month_labels))
    encoding[month_labels.index(label)] = 1
    months_encoded.append(encoding)

final_samples = []
for i in range(0, len(samples)):
    features = []
    features.append(str(samples[i][0]))
    features.append(str(samples[i][1]))
    features.append(str(location_norm[i][0]))
    features.append(str(location_norm[i][1]))
    features += map(str, canon_encoded[i])
    features += map(str, months_encoded[i])
    final_samples.append(features)
            
# Write to all.csv
with open(data_output_path + 'all.csv', 'w+') as output_file:
    writer = csv.DictWriter(output_file, column_names)
    writer.writeheader()
    for row in output:
        writer.writerow(row)
        
# Write to all_features.csv
with open(data_output_path + 'all_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in final_samples:
        writer.writerow(row)
        
# Split into training and test
split_index = int(len(final_samples) * 0.7)
train_samples = final_samples[:split_index]
test_samples = final_samples[split_index:]

# Write to train_features.csv and test_features.csv
with open(data_output_path + 'train_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in train_samples:
        writer.writerow(row)
        
with open(data_output_path + 'test_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in test_samples:
        writer.writerow(row)