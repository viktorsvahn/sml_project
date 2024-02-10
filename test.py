#!/usr/bin/python

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES/OBJECTS ###################################################
TRAIN_RATIO = 0.8

column_norm_selector = {
	'near_fid':None, 	# categorical, should this be included?
	'near_angle':'minmax',	# Uniform on x \in [-180,180]
	'log_dist':'minmax',	# Two peaks, seemingly normally distributed
	'age':'minmax',			# Almost uniform on x \in [20,80]
	'heard':None,			# label: N/A
	'building':None,		# bool
	'noise':None,			# bool
	'in_vehicle':None,		# bool
	'no_windows':None,		# bool
#	'asleep':None,			# bool
}





# Fixed random seed for reproducibility
np.random.seed(123456)


# FUNCTIONS ###################################################################
# Data
def feature_rescale(df, column_info):
	"""Given a dataframe and a column->norm-type map, returns a new dataframe
	with features rescaled accordingly."""
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

		elif norm_type == 'root':
			xnorm = np.sqrt(x)

		elif norm_type == 'stdev':
			mu = np.mean(x)
			sdtev = np.std(x)
			xnorm = (x-mu)/sdtev

		elif norm_type == None:
			xnorm = x

		else:
			print('No standardisation scheme selected.')

		temp_df[col] = xnorm
	return temp_df

# Tree grow
def entropy(frequencies):
	"""Returns the Shannon entropy over some tuple/list of frequencies."""
	return np.sum([-pi*np.log(pi) for pi in frequencies])

def relative_occurrence(arg):
	"""Determines the relative frequency of a given class in a categorical
	array."""
	classes, counts = np.unique(arg, return_counts=True)
	return counts/sum(counts)

def evaluate_gain(*args):
	"""Evaluates the information gain across a set of lists."""
	gain = 0
	for arg in args:
		pi = relative_occurrence(arg)	# Rel, class fraction
		S = entropy(pi)					# Entropy
		EA = sum(pi*S/NUM_ROWS)			# Expected average information
		gain += S - EA					# Information gain
	return gain

def split(arr, idx):
	"""Given some array and index, split array at index and return both as a
	list."""
	return [arr[:idx], arr[idx:]]


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

TOTAL_ENTROPY = entropy([NUM_NOT_HEARD/NUM_ROWS, NUM_HEARD/NUM_ROWS])
print(TOTAL_ENTROPY)


# PRE-PROCESSING ##############################################################

# Filter data
df = df[df['asleep'] == 0] # Filters to non-sleeping instances

# Convert coords to log-distances between individual and fid
dist_x = df.pop('near_x') - df.pop('xcoor')
dist_y = df.pop('near_y') - df.pop('ycoor')
dist = np.sqrt(dist_x**2 + dist_y**2)
df['log_dist'] = np.log(dist)

# Select columns included in the preamble
df = df[column_norm_selector.keys()]
print(df.columns)

# SHUFFLE AND SPLIT
# Shuffle all rows. Frac is ratio of rows to return, randomly.
df = df.sample(frac=1)
#print(df)

# Extract and split labels
labels = df.pop('heard')
train_labels = labels[:NUM_TRAIN]
test_labels = labels[NUM_TRAIN:]

# Extract and split features
train_df = df[:NUM_TRAIN]
test_df = df[NUM_TRAIN:]

#print(train_df, train_labels)
#print(test_df, test_labels)


# feature_rescale AND ZERO-CENTRE FEATURES 
train_df = feature_rescale(train_df, column_norm_selector)
test_df = feature_rescale(test_df, column_norm_selector)
#print(train_df, train_labels)


"""
Loss function: shannon entropy?

Cost funciton is used to evaluate splits in the tree

Splits require one input attribute and one associated label.

The cost function measures the goodness of a split. 

SHOULD THIS BE GENERALISED FOR MORE THAN BINARY SPLITS?
HOW DO WE GENERALISE FOR MILTIPLE ATTRIBUTES?
"""



def optimise_split(X, Y):
	"""Optimises the split of X based on the associated information gain of
	making the split on Y."""
	classes, counts = np.unique(Y, return_counts=True)	
	NUM_IDS = len(X)
	
	gains = []
	for i in range(NUM_IDS):
		XX, YY = split(X, i), split(Y, i)
		gains.append(evaluate_gain(*YY))
	s = np.argmax(gains)

	return split(X, s), split(Y, s)


def grow(maxit, Y, *args):
	pass
	frac = 0.5
	#print(X)
	leaves = {}
	# optimise split for all elements in list only if criteria is met
	# Criteria can be a maximum number of steps and/or a given characteristic of a list
	# Examples of charateristics is to stop on pure nodes. Stop should mean that we return
	# the counts within the leaf.
	# IDEA: Create a dict where only leaves are added.
	#       IF criteria not met, add to dict. When max number of iterations
	#       have been reached, add all remaining lists (leaves) to the dict.
	
	Xsplit, Ysplit = optimise_split(*args, Y)
	print(Xsplit, Ysplit)
	"""
	Plan:
	  1. Generate an optimal split for the root
	  2. For each split, if the frequency of one class is higher than 'frac',
		 add the associated split associated pi-list (using pop) for it to leaves[n]
	  3. Otherwise, call 'Xsplit, Ysplit = optimise_split(X, Y)' again
	  4. Repeat until maxit is reached. Return leaves.
		
		Note: we want to split the categories apart, not the values themselves.
		Pure splits have high entropy.
		Each splitr should have associated with it, a tuple of the relative class
		counts.

		Randomly select attribute each split?

	  In principle (according to nature article), what i am doing is correct.
	  for X<x or X>x, simply take X[s]
	  How determine direction of inequality?
	"""
	for n in range(maxit):

		for y in Ysplit:
			pass
			#print(y)
			#if len(l) == 2:
			#	leaves[n] = Ysplit.pop(l)
	#print(leaves)

	return leaves


# Control parameters
N = 10
depth = 1

# Generate data
X = np.arange(N)
Y = np.random.randint(0,2, N)
T = len(Y[Y == 1])
F = len(Y[Y == 0])

#print(X, Y)
#print(T, F)


# Split
#idx = 5 # The parameter that must be optimised in each step

#XY = optimise_split(X, Y)
tree = grow(depth, Y, X)

#X1, X2 = X[:idx], X[idx:]
#Y1, Y2 = Y[:idx], Y[idx:]

#print(X1, X2)
#print(Y1, Y2)




# Evaluate
#g = evaluate_gain(Y1, Y2)
#print(g)

#print(S1, EA1, gain1)
#print(S2, EA2, gain2)






















A = 2
N = 10
depth = 2

# Generate data
X = np.arange(A*N)
np.random.shuffle(X)
X = X.reshape(N,A)
Y = np.random.randint(0,2, N)

df = pd.DataFrame(X, columns=[f'A{i}' for i in range(A)])
df['Y'] = Y
#print(df)



print(79*'#')


attribute_type = {
	'near_fid':'Categorical', 	# categorical, should this be included?
	'near_angle':'Numerical',	# Uniform on x \in [-180,180]
	'log_dist':'Numerical',	# Two peaks, seemingly normally distributed
	'age':'Numerical',			# Almost uniform on x \in [20,80]
	'heard':None,			# label: N/A
	'building':bool,		# bool
	'noise':bool,			# bool
	'in_vehicle':bool,		# bool
	'no_windows':bool,		# bool
#	'asleep':bool,			# bool
}

attribute_type = {
	'A0':'Numerical',
	'A1':bool,
	'A2':bool,
	'A3':bool,
	'A4':bool,
}


class Tree():
	"""Instantiation of this class leads to the creation of a node that
	immediately splits into two, if some condition is met.
	"""
	leaf_list = list()
	leaves = dict()
	#non_terminal_splits = dict()
	#left = dict()
	#right = dict()
	tmp_gains = []
	gains = dict()
	node_count = 0

	def __init__(self, X, Y, MAX_DEPTH=None):
		self.X = X
		self.Y = Y
		if MAX_DEPTH is not None:
			Tree.MAX_DEPTH = MAX_DEPTH				# Set max-depth on init
		
		self.split_condition = None					# (attr, split_index) 
		
		if Tree.node_count == 1:
			self.NUM_ROWS = len(self.Y)
		Tree.node_count += 1
		


	# Primary methods
	def grow(self, NUM_PARENTS=1):
		""" """
		# Attributes that are 0,1 needs to be converted to booleans
		self.attributes = self.X.columns
		rand_attr = np.random.choice(self.attributes)
		random_attribute = self.X[rand_attr]
		attr_type = attribute_type[rand_attr]
		if attr_type == bool: random_attribute.astype(bool)

		# Split condition
		s, self.gain =  self.optimise_split(
			random_attribute,
			self.Y
		)
		Tree.gains[NUM_PARENTS-1] = self.gain
		#Tree.tmp_gains.append(self.gain)

		self.split_condition = (rand_attr, random_attribute[s])
		#print(self.split_condition)

		if NUM_PARENTS < Tree.MAX_DEPTH:

			Tree.gains[NUM_PARENTS] = Tree.tmp_gains
			NUM_PARENTS += 1
			left_X_split, right_X_split = self.split(self.X, s)
			left_Y_split, right_Y_split = self.split(self.Y, s)
			self.split_left = Tree(left_X_split, left_Y_split)
			self.split_left.grow(NUM_PARENTS)
			
			self.split_right = Tree(right_X_split, right_Y_split)
			self.split_right.grow(NUM_PARENTS)

		else:
			rel_freq = self.relative_occurrence(self.Y)
			Tree.leaf_list.append(rel_freq)
			Tree.leaves[NUM_PARENTS] = Tree.leaf_list
		
	def classify(self):
		"""Left splits are 'greater than' for numerical values and 'True' for 
		boolean. The reverse is true for right splits."""
		pass


	# Auxillary methods
	def entropy(self, frequencies):
		"""Returns the Shannon entropy over some tuple/list of frequencies."""
		return np.sum([-pi*np.log(pi) for pi in frequencies])

	def relative_occurrence(self, arg):
		"""Determines the relative frequency of a given class in a categorical
		array."""
		classes, counts = np.unique(arg, return_counts=True)
		#print(classes, counts)
		return counts/sum(counts)

	def evaluate_gain(self, *args):
		"""Evaluates the information gain of a split."""
		gain = 0								# Total information gain of split
		for arg in args:
			pi = self.relative_occurrence(arg)	# Relative class occurrence
			S = self.entropy(pi)				# Shannon entropy of left/right
			EA = sum(pi*S/NUM_ROWS)				# Expected avg. info of left/right
			gain += S - EA						# Information gain of left/right
		return gain

	def split(self, df, idx):
		"""Given some array and index, split array at index and return both as a
		list."""
		return [df.iloc[:idx], df.iloc[idx:]]

	def optimise_split(self, X, Y):
		"""Optimises the split of X based on the associated information gain of
		making the split on Y."""
		classes, counts = np.unique(Y, return_counts=True)	
		NUM_IDS = len(X)
		gains = []
		for i in range(NUM_IDS):
			XX, YY = self.split(X, i), split(Y, i)
			gain = self.evaluate_gain(*YY)
			gains.append(gain)
		s = np.argmax(gains)	
		MAX_GAIN = max(gains)
		
		return s, MAX_GAIN


Y = df.pop('Y')
#print(df)
#print(Y)



tree = Tree(df, Y, depth)
tree.grow()

#print(tree.X)
#print(tree.split_right.X)
#print(tree.split_left.X)

print('Number of nodes:', tree.node_count)
print('Gains:', tree.gains)
print('Leaves:', tree.leaves)

