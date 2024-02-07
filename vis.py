#!/usr/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_RATIO = 0.8

df = pd.read_csv('siren_data_train.csv')
print(df.columns)

NUM_ROWS = len(df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)
NUM_TEST = NUM_ROWS-NUM_TRAIN


# PRE-PROCESSING
# Remove sleeping people
#df = df[df['asleep'] == 0]
print(df)

labels = df.pop('heard')

fig, axs = plt.subplots(4,3, figsize=(8,8))
axs = axs.ravel()

for i, col in enumerate(df.columns):
	ax = axs[i]

	ax.set_xlabel(col)
	ax.hist(df[col], bins=25, density=True)


plt.tight_layout()
plt.show()