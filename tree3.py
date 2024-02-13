#!/usr/bin/python

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


attribute_type = {
	#'near_fid':'Categorical', 	# categorical, should this be included?
	'near_angle':'Numerical',	# Uniform on x \in [-180,180]
	'log_dist':'Numerical',	# Two peaks, seemingly normally distributed
	'age':'Numerical',			# Almost uniform on x \in [20,80]
	'heard':None,			# label: N/A
	'building':bool,		# bool
	'noise':bool,			# bool
	'in_vehicle':bool,		# bool
	'no_windows':bool,		# bool
	'asleep':bool,			# bool
}

# FUNCTIONS ###################################################################
class Node:
	node_count = 1
	def __init__(self,
		parent=None,
		left=None, 
		right=None
	):
		self.parent = parent 						# Parent (attribute, entropy/gain)
		self.left = left 							# Greater than/True node object
		self.right = right 							# Less than/False node object

		Node.node_count += 1


class Tree(Node):
	tmp_gains = [] 									# For statistics
	gains = dict()									# For statistics

	def __init__(self,
		data,
		label,
		MAX_DEPTH=None,
		*args
	):
		super().__init__(*args)
		self.X = data
		self.label = label
		
		self.NUM_SAMPLES = len(self.X)
		self.attributes = self.X.columns


		# Shannon entropy of entire dataset
		self.pi = self.relative_occurrence(self.X[self.label])
		self.dataset_entropy = self.shannon_entropy(self.pi)

		print('dataset entropy =', self.dataset_entropy)

		if MAX_DEPTH is None:
			#self.grow(self.NUM_PARENT)
			pass
		else:
			Tree.MAX_DEPTH = MAX_DEPTH				# Stopping criterion
			
			# Reset tree by including MAX_DEPTH in instantiation
			Tree.tmp_gains = [] 					# For statistics
			Tree.gains = dict()						# For statistics
			Node.node_count = 1 					# Should be 2^MAX_DEPTH-1 at most
		
			#self.grow(1)


		#print(self.dataset_entropy)
		
		# Slush printstatements, delete when done
		#print(self.X)
		#print(self.Y)
		#self.pi = self.relative_occurrence(self.Y)
		#print(self.pi)
		#c = self.classify(self.X['log_dist'], self.Y)
		#o = self.optimise_split(self.X['log_dist'], self.Y)
		#print(o)

	def grow(self, NUM_PARENTS=1):
		pass
		
		if NUM_PARENTS < Tree.MAX_DEPTH:
			#print('asashaskhasljdhasldjhsad')	

			#CATEGORICAL COLUMNS ARE DROPPED, REAL NUMERICAL ARE NOT
			# NUMERICAL SHOULD ALSO BE SORTED AND SPLIT AS PREVIOUS
			

			split_condition, split = self.optimise_split(self.X, self.label)
			#print(split)
			#try:
			#	left, right = [s[1] for s in split]
			#except:
			#	left, right = split
			try:
				left, right = split
			except:
				pass
			#print(left[1].columns)
			#print(right)
			#print(left)
			
			if left is not None:
				print('assadasdsds', left)
				self.left = Tree(left, self.label)
				print(self.left)
				self.left.parent = split_condition
				self.left.grow(NUM_PARENTS+1)
			
			if right is not None:
				print('jhjhghgjgh', right)
				self.right = Tree(right, self.label)
				self.right.parent = split_condition
				self.right.grow(NUM_PARENTS+1)

		else:
			pass


	# Auxiliary methods	
	def relative_occurrence(self, arr):
		"""Determines the relative frequency of a given class in a categorical
		array."""
		_, counts = np.unique(arr, return_counts=True)
		#print(classes, counts)
		return counts/sum(counts)

	def shannon_entropy(self, frequencies):
		"""Returns the Shannon entropy associated with a set of class 
		frequencies."""
		return np.sum([-pi*np.log2(pi) for pi in frequencies])

	def info(self, classifications):
		"""Determines the information gain (IGain) associated with splitting a
		parent node into len(*Y) branches.

		  Returns:
		    IGain = Shannon_entropy(parent_df) 
		            - ( |left_df|/|parent_df|*Shannon_entropy(left) 
		              + |right_df|/|parent_df|*Shannon_entropy(right) )
		"""
		RCOUNT = classifications.value_counts(normalize=True)
		ENTROPY = self.shannon_entropy(RCOUNT)
		INFO = ENTROPY*len(classifications)/self.NUM_SAMPLES
		return INFO

	def split(self, df, split_index):
		"""Given a dataframe (df) and an index (s), return a list of two 
		disjoint dataframes that have been split at the given index."""
		return [df.iloc[:split_index], df.iloc[split_index:]]

	def optimise_split(self, data, label):
		"""Optimises the split of Y based on the associated maximum information
		gain (IGain) of making the split.

		  Returns:
		    - Index that maximises IGain, and
		    - its associated Igain"""
		
		gains = {}
		for attribute in self.attributes:
			if attribute != label:				
				attribute_realisations = data[attribute]
				
				if attribute_realisations.dtype == bool:	
					# Without lambda, iteration of split returns a tuple of
					# value and dataframe
					splits = data.groupby(attribute).apply(lambda x: x)
					GAIN = 0
					for s in splits:
						split = s[1]
						if len(split) > 2:
							split.pop(attribute)							
							classifications = split[label] # classifications
							GAIN += self.dataset_entropy - self.info(classifications)		
					gains[GAIN] = [(attribute, data[attribute][index]), splits]

				else:
					data = data.sort_values(attribute).reset_index(drop=True)

					for index in data.index:
						if (index > 0) and (index < max(data.index)):
							splits = self.split(data, index)							
							INFO = 0
							for s in splits:
								INFO += self.info(s[label])						
							# Order of splits is reveresed to have left output
							# always be greater than for numerical attributes
							GAIN = self.dataset_entropy - INFO
							gains[GAIN] = [(attribute, data[attribute][index]), splits[::-1]]

		optimal_split = gains[max(gains)]
		return optimal_split



# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.002
depth = 4


# Fixed random seed for reproducibility
np.random.seed(12345)


# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')

# Variables
NUM_ROWS = len(df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)
NUM_TEST = NUM_ROWS-NUM_TRAIN

NUM_HEARD = len(df[df['heard'] == 1])
NUM_NOT_HEARD = NUM_ROWS - NUM_HEARD
print(NUM_HEARD, NUM_NOT_HEARD)


# PRE-PROCESSING ##############################################################

# Filter data

df.pop('near_fid')
#df = df[df['asleep'] == 0] # Filters to non-sleeping instances

# Convert coords to log-distances between individual and fid
dist_x = df.pop('near_x') - df.pop('xcoor')
dist_y = df.pop('near_y') - df.pop('ycoor')
dist = np.sqrt(dist_x**2 + dist_y**2)
df['log_dist'] = np.log(dist)

# Select columns included in the preamble
print(df.columns)







#tree = Tree(1,2,3,4,5,6)
#tree = Tree(1,2,3)
#print(tree)
#print(tree.X, tree.Y, tree.MAX_DEPTH)
#print(tree.parent, tree.left, tree.right)









# Express booleans as {True,False} instead of {0,1}
attribute_type = {
	'near_fid':'Categorical', 	# categorical, should this be included?
	'near_angle':'Numerical',	# Uniform on x \in [-180,180]
	'log_dist':'Numerical',	# Two peaks, seemingly normally distributed
	'age':'Numerical',			# Almost uniform on x \in [20,80]
	'heard':bool,			# label: N/A
	'building':bool,		# bool
	'noise':bool,			# bool
	'in_vehicle':bool,		# bool
	'no_windows':bool,		# bool
	'asleep':bool,			# bool
}

for attribute in df:
	if attribute_type[attribute] == bool:
		df[attribute] = df[attribute].astype(bool)

#print(df)


runs = 1
gns = []
depths = np.arange(depth)

print('Running...')
for run in range(runs):
	#np.random.seed(run)
	tmp_df = copy.deepcopy(df)

	tmp_df = tmp_df.sample(frac=1).reset_index(drop=True)
	#print(df)

	# Extract and split labels
	#labels = tmp_df.pop('heard')
	#train_labels = labels[:NUM_TRAIN]
	#test_labels = labels[NUM_TRAIN:]

	# Extract and split features
	train_df = tmp_df[:NUM_TRAIN]
	test_df = tmp_df[NUM_TRAIN:]

	tree = Tree(train_df, 'heard', depth)
	tree.grow()
	
	print(tree.node_count)
	#print(tree)
	#print(tree.gains)
	#gns.append(list(tree.gains.values()))
	
print('Finished!')
#print(tree.left.X)

#print(gns)
#avg_gns = np.mean(gns, axis=0)
#print(avg_gns)

#std_gns = np.std(gns, axis=0)
#print(std_gns)

#for g in gns:
#	plt.plot(depths, g, '.-')
#plt.errorbar(depths, avg_gns, yerr=std_gns) 



#plt.show()
