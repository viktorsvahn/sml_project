#!/usr/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################


# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')



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
fig, ax = plt.subplots(figsize=(3.3,3.3))

NBINS = 50
ax.set_title(f'NUM_BINS={NBINS}')
ax.set_xlabel('$\\log(\\text{dist})$')
ax.hist(np.log(df['dist']), bins=NBINS, density=True)


plt.tight_layout()
plt.show()