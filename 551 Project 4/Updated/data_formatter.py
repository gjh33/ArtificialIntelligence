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
from sklearn import preprocessing
import pickle
from datetime import datetime, timedelta
from math import degrees, radians, cos, sin, asin, sqrt, atan2

#####CONFIGURATION#####
# Files to combine
file_names = ['bald_eagle_1.csv', 'bald_eagle_2.csv', 'buzz_eagle.csv']
# Path to folder containing these files. Don't forget to escape backslashes.
#raw_data_path = 'C:\\Users\\THEANO\\Documents\\Comp 551\\Project 4\\raw\\'
raw_data_path = 'raw/'
# Path to output folder for combined file
#data_output_path = 'C:\\Users\\THEANO\\Documents\\Comp 551\\Project 4\\data\\'
data_output_path = 'data/'
# Columns to load. Make sure all files have these columns
column_names = ['event-id', 'timestamp', 'location-long', 'location-lat', \
                'individual-taxon-canonical-name', 'tag-local-identifier']
#####END CONFIGURATION#####

#####FUNCTIONS#####
def parseTime(time):
    timeFormat = '%Y-%m-%d %H:%M:%S.%f'
    dt = datetime.strptime(time, timeFormat)
    return dt

def timeDif(current, last):
    # 2014-12-31 19:00:00.000
    tDelta = parseTime(current) - parseTime(last)
    return tDelta.seconds / 3600

def formatTime(samples):
    lastTime = {}
    for sample in samples:
        if sample[1] not in lastTime: # Identifier not found
            lastTime[sample[1]] = sample[2] # Store last time
            sample[2] = 0
            continue
        newLastTime = sample[2]
        sample[2] = timeDif(sample[2], lastTime[sample[1]])
        lastTime[sample[1]] = newLastTime

def getDistance(loc1, loc2):
    """
    Calculates the haversine distance between two location points.
    """
    R = 6373
    # DD to Radians
    loc1 = loc1.astype(float)
    loc2 = loc2.astype(float)
    lon1, lat1, lon2, lat2 = map(radians, [loc1[0], loc1[1], loc2[0], loc2[1]])
    # Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    #c = 2 * atan2(sqrt(a), sqrt(1-a))
    c = 2 * asin(sqrt(a))
    distInKm = R * c

    return distInKm

def getBearing(loc1, loc2):
    """
    Calculates the bearing in degrees (0 to 360) from loc1 to loc2.
    """
    # DD to Radians
    loc1 = loc1.astype(float)
    loc2 = loc2.astype(float)
    lon1, lat1, lon2, lat2 = map(radians, [loc1[0], loc1[1], loc2[0], loc2[1]])
    # Bearing
    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
    bearing = atan2(y, x)
    # Convert to compass bearing
    bearing = (degrees(bearing) + 360) % 360

    return bearing

def addDistanceAndBearing(samples):
    lastLoc = {}
    initial = {}
    i = 0
    for sample in samples:
        if sample[1] not in lastLoc: # Identifier not found
            lastLoc[sample[1]] = sample[3:5] # Store last location
            initial[sample[1]] = i, None # Store the initial sample index
            i += 1
            continue
        newLoc = sample[3:5]
        sample[5] = getDistance(lastLoc[sample[1]], sample[3:5])
        sample[6] = getBearing(lastLoc[sample[1]], sample[3:5])
        lastLoc[sample[1]] = newLoc
        if (initial[sample[1]][1] is None):
            initial[sample[1]] = initial[sample[1]][0], sample
        i += 1

    for label, values in initial.items():
        i, sample = values
        samples[i][6] = sample[6] # Change initial bearing
#####END FUNCTIONS#####

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
#output = sorted(output, key=lambda k: int(k[column_names[0]])) # Sorted by event-id
output = sorted(output, key=lambda k: parseTime(k[column_names[1]])) # Sorted by time

# Convert into feature mapping
samples = []
for sample in output:
    feature_list = []
    feature_list.append(int(sample['event-id']))
    feature_list.append(int(sample['tag-local-identifier']))
    feature_list.append(sample['timestamp']) # Adding time
    feature_list.append(float(sample['location-long']))
    feature_list.append(float(sample['location-lat']))
    # Add distance + bearing
    feature_list.append(0.) # Distance
    feature_list.append(0.) # Bearing
    feature_list.append(sample['individual-taxon-canonical-name'])
    samples.append(feature_list)

# Convert to numpy array
samples = numpy.array(samples)
print("Formatting time...")
formatTime(samples)
print("Done.")
print("Adding distance and bearing...")
addDistanceAndBearing(samples)
print("Done.")

scaler = preprocessing.MinMaxScaler() # Feature scaler
distance_norm = scaler.fit_transform(samples[:, 5]) # Throws deprecation warning
bearing_norm = scaler.fit_transform(samples[:, 6])
location_norm = scaler.fit_transform(samples[:, 3:5].tolist()) # Scale location features
with open('scaler.pkl', 'wb') as pickle_file:
    pickle.dump(scaler, pickle_file)
canon_name_col = samples[:, [7]].flatten() # Get species name
canon_labels = numpy.unique(canon_name_col) # Get unique species name
canon_labels = canon_labels.tolist()
canon_encoded = []
for label in canon_name_col: # For every species
    encoding = numpy.zeros(len(canon_labels))
    encoding[canon_labels.index(label)] = 1
    canon_encoded.append(encoding)

final_samples = []
for i in range(0, len(samples)):
    features = []
    features.append(str(samples[i][0]))
    features.append(str(samples[i][1]))
    features.append(str(samples[i][2]))
    features.append(str(location_norm[i][0]))
    features.append(str(location_norm[i][1]))
    #features.append(str(samples[i][3]))
    #features.append(str(samples[i][4]))
    features.append(str(distance_norm[i]))
    features.append(str(bearing_norm[i]))
    #features.append(str(samples[i][5]))
    #features.append(str(samples[i][6]))
    features += map(str, canon_encoded[i]) # 1-hot encoding
    final_samples.append(features)

# Write to all.csv
print("Writing to all.csv...")
with open(data_output_path + 'all.csv', 'w+') as output_file:
    writer = csv.DictWriter(output_file, column_names)
    writer.writeheader()
    for row in output:
        writer.writerow(row)
print("Done.")

# Write to all_features.csv
print("Writing to all_features.csv...")
with open(data_output_path + 'all_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in final_samples:
        writer.writerow(row)
print("Done.")

# Split into training and test
split_index = int(len(final_samples) * 0.7)
train_samples = final_samples[:split_index]
test_samples = final_samples[split_index:]

# Write to train_features.csv and test_features.csv
print("Writing to train_features.csv and test_features.csv...")
with open(data_output_path + 'train_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in train_samples:
        writer.writerow(row)

with open(data_output_path + 'test_features.csv', 'w+') as output_file:
    writer = csv.writer(output_file)
    for row in test_samples:
        writer.writerow(row)
print("Done.")
