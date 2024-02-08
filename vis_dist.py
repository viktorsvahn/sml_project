#!/usr/bin/python

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################


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
ids = df.pop('near_fid')

dist_x = df.pop('near_x') - df.pop('xcoor')
dist_y = df.pop('near_y') - df.pop('ycoor')
dist = np.sqrt(dist_x**2 + dist_y**2)
log_dist = np.log(dist)
maxlogdist = max(log_dist)
minlogdist = min(log_dist)
minmax_log_dist = 2*((log_dist-minlogdist)/(maxlogdist - minlogdist)) -1
#print(df)
df['log_dist'] = log_dist
df['minmax_log_dist'] = minmax_log_dist
#print(df)
print(df.columns)

l = ['log_dist', 'minmax_log_dist']

# PLOTTING ####################################################################
fig, axs = plt.subplots(1,2, figsize=(6.6,3.3))
axs = axs.ravel()

for i, ax in enumerate(axs):

	ax.set_title(f'NUM_BINS={NBINS}')
	if i == 0:
		ax.set_xlabel('$\\log(\\text{dist})$')
	else:
		ax.set_xlabel('Scaled $\\log(\\text{dist})$')
	ax.hist(df[l[i]], bins=NBINS, density=False)


plt.tight_layout()
plt.show()