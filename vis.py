#!/usr/bin/python

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.8

column_bin_selector = {
	'near_fid':50,
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

# Bin width selection
#NBINS = column_bin_selector[col]		# Manually set widths
#if NBINS == None: NBINS = 10			# Pertains to the above
#NBINS = math.ceil(np.log(NUM_ROWS)+1)	# Sturge's formula
NBINS = math.ceil(2*NUM_ROWS**(1/3))	# Rice rule
#NBINS = math.ceil(np.sqrt(NUM_ROWS))	# Square-root choice


# PRE-PROCESSING ##############################################################
labels = df.pop('heard')
#ids = df.pop('near_fid')

dist_x = df.pop('near_x') - df.pop('xcoor')
dist_y = df.pop('near_y') - df.pop('ycoor')
dist = np.sqrt(dist_x**2 + dist_y**2)
#print(df)
df['dist'] = dist
#print(df)
print(df.columns)

# PLOTTING ####################################################################
fig, axs = plt.subplots(3,3, figsize=(9.9,9))
axs = axs.ravel()

for i, col in enumerate(df.columns):
	ax = axs[i]

	

	ax.set_title(f'NUM_BINS={NBINS}')
	ax.set_xlabel(col)
	if col == 'dist':
		ax.ticklabel_format(axis='x', scilimits=(5,5), style='sci', useMathText=True)
	ax.hist(df[col], bins=NBINS, density=False)
	#axs[i] = df.hist(col)


plt.tight_layout()
plt.show()