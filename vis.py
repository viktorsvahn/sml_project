#!/usr/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.8

column_bin_selector = {
#	'near_fid':50,
	'dist':50,
	'near_angle':50,
#	'heard':None,
	'building':None,
	'noise':None,
	'in_vehicle':None,
	'asleep':None,
	'no_windows':None,
	'age':70,
}

# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')

# Variables
NUM_ROWS = len(df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)
NUM_TEST = NUM_ROWS-NUM_TRAIN


# PRE-PROCESSING ##############################################################
labels = df.pop('heard')
ids = df.pop('near_fid')

dist_x = df.pop('near_x') - df.pop('xcoor')
dist_y = df.pop('near_y') - df.pop('ycoor')
dist = np.sqrt(dist_x**2 + dist_y**2)
#print(df)
df['dist'] = dist
#print(df)
print(df.columns)

# PLOTTING ####################################################################
fig, axs = plt.subplots(2,4, figsize=(11,5))
axs = axs.ravel()

for i, col in enumerate(df.columns):
	ax = axs[i]
	nbins = column_bin_selector[col]
	if nbins == None: nbins = 10
	ax.set_title(f'NUM_BINS={nbins}')
	ax.set_xlabel(col)
	ax.hist(df[col], bins=nbins, density=True)


plt.tight_layout()
plt.show()