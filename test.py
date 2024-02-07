#!/usr/bin/python

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.8

column_norm_selector = {
	'near_fid':'minmax',
	'near_x':'minmax',
	'near_y':'minmax',
	'near_angle':'minmax',
	'heard':'minmax',
	'building':None,
	'xcoor':'minmax',
	'ycoor':'minmax',
	'noise':None,
	'in_vehicle':None,
#	'asleep':None,
	'no_windows':None,
	'age':'minmax',
}

# Fixed random seed for reproducibility
np.random.seed(123456)


# FUNCTIONS ###################################################################
def normalise(df, column_info):
	"""Given a dataframe and a column->norm-type map, returns a new dataframe
	with features normalised accordingly."""
	temp_df = copy.deepcopy(df)
	
	for i, col in enumerate(temp_df):
		x = temp_df[col]
		norm_type = column_info[col]

		if norm_type == 'minmax':
			xmin = np.min(x)
			xmax = np.max(x)
			xnorm = 2*((x-xmin)/(xmax-xmin)) - 1

		elif norm_type == 'log':
			xnorm = np.log(x)

		elif norm_type == 'exp':
			xnorm = np.exp(x)

		elif norm_type == 'stdev':
			mu = np.mean(x)
			sdtev = np.std(x)
			xnorm = (x-mu)/sdtev

		elif norm_type == None:
			xnorm = x

		else:
			print('No normalisation scheme selected.')

		temp_df[col] = xnorm
	return temp_df


# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')

# Variables
NUM_ROWS = len(df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)
NUM_TEST = NUM_ROWS-NUM_TRAIN


# PRE-PROCESSING ##############################################################
# Filter data
df = df[df['asleep'] == 0] # Filters to non-sleeping instances
df = df[column_norm_selector.keys()]
print(df.columns)


# SHUFFLING
# Shuffle all rows. Frac is ratio of rows to return, randomly.
df = df.sample(frac=1)
#print(df)


# SPLITTING
# Label data
labels = df.pop('heard')
train_labels = labels[:NUM_TRAIN]
test_labels = labels[NUM_TRAIN:]

# Features
train_df = df[:NUM_TRAIN]
test_df = df[NUM_TRAIN:]

#print(train_df, train_labels)
#print(test_df, test_labels)


# NORMALISE AND ZERO-CENTRE FEATURES 
train_df = normalise(train_df, column_norm_selector)
test_df = normalise(test_df, column_norm_selector)
#print(train_df, train_labels)
