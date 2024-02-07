#!/usr/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.8

column_bin_selector = {
	'near_fid':50,
	'near_x':50,
	'near_y':50,
	'near_angle':50,
	'heard':None,
	'building':None,
	'xcoor':50,
	'ycoor':50,
	'noise':None,
	'in_vehicle':None,
	'asleep':None,
	'no_windows':None,
	'age':70,
}

# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')
print(df.columns)

# Variables
NUM_ROWS = len(df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)
NUM_TEST = NUM_ROWS-NUM_TRAIN


# PRE-PROCESSING ##############################################################
labels = df.pop('heard')

# PLOTTING ####################################################################
fig, axs = plt.subplots(4,3, figsize=(8,8))
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