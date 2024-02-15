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
	node_count = 0
	leaf_count = 0
	def __init__(self,
		parent=None,
		left=None, 
		right=None,
		leaf=None,
	):
		self.parent = parent
		self.left = left
		self.right = right
		self.leaf = leaf


class Tree(Node):
	"""Implementaion of a variant of a C4.5 classification tree algorithm by 
	Breiman (1984) and Quinlan (1986).

	This version makes use of unnomralised, rather than normalised, information 
	gain."""
	#tmp_gains = []
	gains = {}

	def __init__(self,
		data,
		label,
		MAX_DEPTH=None,
		FOREST_RATE=1,
		*args
	):
		super().__init__(*args)
		self.X = data
		self.label = label
		self.FOREST_RATE = FOREST_RATE
		
		self.NUM_SAMPLES = len(self.X)
		self.attributes = self.X.columns

		# Shannon entropy of entire dataset
		self.pi = self.relative_occurrence(self.X[self.label])
		self.dataset_entropy = self.shannon_entropy(self.pi)
		#print('dataset entropy =', self.dataset_entropy)

		if MAX_DEPTH is None:
			pass
		else:
			Tree.MAX_DEPTH = MAX_DEPTH

			# Reset tree by including MAX_DEPTH in instantiation
			Tree.tmp_gains = []
			Tree.gains = {0:self.dataset_entropy}
			Node.node_count = 0
			Node.leaf_count = 0

	def grow(self, NUM_PARENTS=0):
		"""Tree growth method. Calling this wihtout an argument will initiate a
		the growth of a new tree. To generate a tree, call:

			tree = Tree(train_data, 'label', max_depth)
			tree.grow()

		where the label is the name of the column of train_data that contains 
		the classification values. Max depth specifies the maximum possible depth
		of the tree, unless it self terminates before.

		Assuming the max depth has not been reached, leaves are only created if 
		nodes are pure, otherwise it will split."""
		#print(NUM_PARENTS, Tree.MAX_DEPTH)
		if (NUM_PARENTS <= Tree.MAX_DEPTH):

			# Leaves can not be created at first node
			if NUM_PARENTS > 0:
				labels = self.X[self.label]
				NUM_LABELS = len(labels)
				
				# Dominating labels
				MAX_CLASS = labels.max()

				# Label counts
				class_counts = labels.value_counts()
				
				# Label at max count
				CLASSIFICATION = class_counts.idxmax()
				
				if class_counts[CLASSIFICATION] == NUM_LABELS:
					self.leaf = CLASSIFICATION
					Node.leaf_count += 1

			# If current object is not a leaf, try split
			if self.leaf is None:
				condition, split, GAIN = self.optimise_split(self.X, self.label)
				Tree.gains[NUM_PARENTS] = GAIN
				try:
					left, right = split
					self.parent = condition
					self.left = Tree(left, self.label)
					self.left.grow(NUM_PARENTS+1)
					
					self.right = Tree(right, self.label)
					self.right.grow(NUM_PARENTS+1)
					Node.node_count += 1

				# If no optimal split possible, create leaf
				except:
					self.leaf = CLASSIFICATION
					Node.leaf_count += 1

		# Convert to leaf upon reaching maximum depth
		elif NUM_PARENTS-1 == Tree.MAX_DEPTH:
			self.leaf = CLASSIFICATION
			Node.leaf_count += 1
		#if self.leaf is not None: print('leaf: ', self.leaf)


	# Auxiliary methods	
	def relative_occurrence(self, arr):
		"""Determines the relative frequency of a given class in a categorical
		array."""
		_, counts = np.unique(arr, return_counts=True)
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
		data = data.sample(frac=self.FOREST_RATE)
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
							split.pop(attribute), data.pop(attribute)
							classifications = split[label]
							GAIN += self.dataset_entropy - self.info(classifications)		
					gains[GAIN] = [
						(attribute, data[attribute][index]),
						splits
					]

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
							
							gains[GAIN] = [
								(attribute, data[attribute][index]),
								splits[::-1], 
								GAIN
							]

		MAX_GAIN = max(gains)
		optimal_split = gains[MAX_GAIN]
		return optimal_split

	def predict(self, test_data, label):
		"""Makes a prediciton on test data based on previous training. The label
		specifies which column in the dataframe is used for classification."""
		# Reset index and prepare output object
		test_data = test_data.reset_index(drop=True)
		tmp = []

		# Classify each sample in the dataset
		for _ in test_data.iterrows():
			row, sample = _
			
			left_branch = copy.deepcopy(self.left)
			right_branch = copy.deepcopy(self.right)
			split_attribute, condition = self.parent
			
			RUN_PREDICTION = True
			count = -1
			while RUN_PREDICTION:
				count += 1
				s = sample[split_attribute]

				# Numerical attributes
				if (type(s) == float) or (type(s) == int):
					# Left
					if s > condition:
						if count == 0:
							branch = left_branch
						else:
							branch = branch.left
					# Right
					else:
						if count == 0:
							branch = right_branch
						else:
							branch = branch.right

				# Booleans
				else:
					if s:
						if count == 0:
							branch = left_branch
						else:
							branch = branch.left
					# Right
					else:
						if count == 0:
							branch = right_branch
						else:
							branch = branch.right

				# Look for leaf
				if branch.leaf is not None:
					tmp.append(branch.leaf)
					RUN_PREDICTION = False
				else:
					pass

		predictions = pd.Series(tmp)
		return predictions


# CONTROL VARIABLES/OBJECTS ###################################################
# Bagging params
DATA_FRACTION = 0.1
ENSEMBLE_SIZE = 2
FOREST_RATE = 1

# Decision tree params
TRAIN_RATIO = 0.8#0.002
depth = 2
runs = 1
save = False
show = False
gain_v_depth = False
rate_v_depth = False

test = False
perf_v_size = True
perf_v_forest_rate = False
# Fixed random seed for reproducibility
np.random.seed(12345)


# LOAD DATA ###################################################################
# Dataframe
df = pd.read_csv('siren_data_train.csv')


# Variables


#NUM_HEARD = len(df[df['heard'] == 1])
#NUM_NOT_HEARD = NUM_ROWS - NUM_HEARD
#print(NUM_HEARD, NUM_NOT_HEARD)
#df = df.iloc[0:NUM_DATA_POINTS]
#print(df)


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



classifiers = []
results = []



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



def classification_tree(train_data, test_data, label, max_depth, forest_rate):
	pass
	print('Fitting...')
	tree = Tree(train_data, 'heard', max_depth, forest_rate)
	tree.grow()		
	print('Nodes: ',tree.node_count)
	print('Leaves: ', tree.leaf_count)

	print('Evaluating performance...')
	# Confusion matrix 
	predict = tree.predict(train_data, label)
	confusion_matrix = np.zeros((2,2))
	for i, j in zip(test_data[label], predict):
		if (i == True) and (j == True):
			confusion_matrix[0][0] += 1
		elif (i == False) and (j == True):
			confusion_matrix[0][1] += 1
		elif (i == True) and (j == False):
			confusion_matrix[1][0] += 1
		elif (i == False) and (j == False):
			confusion_matrix[1][1] += 1
	MISCLASSIFICATION_RATE = (confusion_matrix[0, 1] + confusion_matrix[1, 0])/np.sum(confusion_matrix)
	ACCURACY = 1 - MISCLASSIFICATION_RATE
	print(f'Misclassification rate:\t{MISCLASSIFICATION_RATE:.4f}')
	print(f'Accuracy:\t\t\t\t{ACCURACY:.4f}')
	return (MISCLASSIFICATION_RATE, ACCURACY), tree



for run in range(runs):
	if runs > 1:
		print('Run: ', run)
	
	#np.random.seed(123)
	tmp_df = copy.deepcopy(df)
	tmp_df = tmp_df.sample(frac=DATA_FRACTION).reset_index(drop=True)
	NUM_ROWS = len(tmp_df.index)
	NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)

	# Extract and split features
	train_df = tmp_df.iloc[:NUM_TRAIN]
	test_df = tmp_df.iloc[NUM_TRAIN:]

	
	if test is False:

			if perf_v_size:
				bags = []
				for _ in range(ENSEMBLE_SIZE):
					tmp_df = copy.deepcopy(df)
					tmp_df = tmp_df.sample(frac=DATA_FRACTION).reset_index(drop=True)
					NUM_ROWS = len(tmp_df.index)
					NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)

					# Extract and split features
					train_df = tmp_df.iloc[:NUM_TRAIN]
					test_df = tmp_df.iloc[NUM_TRAIN:]	
					
					# Fit and classify
					result, tree = classification_tree(train_df, test_df, 'heard', depth, FOREST_RATE)
					bags.append(result)

				bags_df = pd.DataFrame(bags)
				print(bags_df)

			elif perf_v_forest_rate:
				pass
				result, tree = classification_tree(train_df, test_df, 'heard', depth, FOREST_RATE)
			
			else:
				result, tree = classification_tree(train_df, test_df, 'heard', depth, FOREST_RATE)
			
			if gain_v_depth:
				x = tree.gains.keys(); y = tree.gains.values()
				yy.append(list(y))
				plt.plot(x,y, '.', label=f'Run {run}')

print('Finished!')

if gain_v_depth:
	out_df = pd.DataFrame(yy)
	out_df['avg'] = out_df.mean(axis=0)
	plt.plot(out_df.index, out_df['avg'], 'k', label='Average', zorder=0) 


	plt.title(f'Depth optimisation with {TRAIN_RATIO:.0%} of data used for training')
	plt.xlabel('Depth')
	plt.ylabel('Max gain')

	if save:
		out_df.to_csv(f'{TRAIN_RATIO*100:.0%}trainsize_{depth}maxdepth_{runs}runs.txt', sep=' ')

plt.legend()
if show:
	plt.show()