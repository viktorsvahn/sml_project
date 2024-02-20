#!/usr/bin/python

import sys
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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
	#node_count = 1
	#leaf_count = 0
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
		NUM_PARENTS=0,
		*args
	):
		super().__init__(*args)
		self.X = data
		self.label = label
		self.FOREST_RATE = FOREST_RATE
		self.NUM_PARENTS = NUM_PARENTS
		#print(self.NUM_PARENTS)

		self.NUM_SAMPLES = len(self.X)
		self.attributes = self.X.columns
		# Shannon entropy of entire dataset
		self.pi = self.relative_occurrence(self.X[self.label])
		self.parent_entropy = self.shannon_entropy(self.pi)
		self.N = len(self.X.index)
		#print('dataset entropy =', self.parent_entropy)

		if MAX_DEPTH is None:
			pass
		else:
			Tree.MAX_DEPTH = MAX_DEPTH

			# Reset tree by including MAX_DEPTH in instantiation
			Tree.tmp_gains = []
			Tree.gains = {0:self.parent_entropy}
			Node.node_count = 0
			Node.leaf_count = 0

	def grow(self):
		"""Tree growth method. Calling this wihtout an argument will initiate a
		the growth of a new tree. To generate a tree, call:

			tree = Tree(train_data, 'label', max_depth)
			tree.grow()

		where the label is the name of the column of train_data that contains 
		the classification values. Max depth specifies the maximum possible depth
		of the tree, unless it self terminates before.

		Assuming the max depth has not been reached, leaves are only created if 
		nodes are pure, otherwise it will split."""
		
		labels = self.X[self.label]
		NUM_LABELS = len(labels)

		if self.NUM_PARENTS+1 < Tree.MAX_DEPTH:
			test_freq = self.relative_occurrence(self.X[self.label])

			if (test_freq[0] == 1) or (self.N < 2):
				CLASSIFICATION = self.classify(labels)
				self.leaf = CLASSIFICATION
				Node.leaf_count += 1
			else:
				try:
					self.condition, split = self.optimise_split(self.X, self.label)
					
					# Split condition of root node 
					if self.NUM_PARENTS == 0:
						self.parent = self.condition

					left, right = split
					self.left = Tree(left, self.label, NUM_PARENTS=self.NUM_PARENTS+1)
					self.left.parent = self.condition
					self.left.grow()

					self.right = Tree(right, self.label, NUM_PARENTS=self.NUM_PARENTS+1)
					self.right.parent = self.condition
					self.right.grow()
					Node.node_count += 1
				except:
					CLASSIFICATION = self.classify(labels)
					self.leaf = CLASSIFICATION
					Node.leaf_count += 1
		else:
			CLASSIFICATION = self.classify(labels)
			self.leaf = CLASSIFICATION
			Node.leaf_count += 1



	# Auxiliary methods	
	def relative_occurrence(self, arr):
		"""Determines the relative frequency of a given class in a categorical
		array."""
		_, counts = np.unique(arr, return_counts=True)
		return counts/sum(counts)

	def shannon_entropy(self, frequencies):
		"""Returns the Shannon entropy associated with a set of class 
		frequencies."""
		if len(frequencies) > 1:
			return np.sum([-pi*np.log2(pi) for pi in frequencies])
		else:
			return 0

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

	def classify(self, labels):
		pass
		# Dominating labels
		MAX_CLASS = labels.max()

		# Label counts
		class_counts = labels.value_counts()
		
		# Label at max count
		CLASSIFICATION = class_counts.idxmax()
		return CLASSIFICATION#, class_counts

	def optimise_split(self, data, label):
		"""Optimises the split of Y based on the associated maximum information
		gain (IGain) of making the split.

		  Returns:
		    - Index that maximises IGain, and
		    - its associated Igain"""
		data = copy.deepcopy(data)
		gains = {}
		
		tmp_attr = list(data.columns)
		#print(tmp_attr)
		tmp_attr.remove(label)
		#print(tmp_attr)
		d = len(data.columns)
		k = math.ceil(np.sqrt(d))
		random_attributes = list(np.random.choice(tmp_attr, k, replace=False))
		random_attributes.append(label)
		#print(random_attributes)

		data = data[random_attributes]

		for attribute in random_attributes:
			if attribute != label:
				attribute_realisations = data[attribute]
				#print(attribute_realisations)
				if attribute_realisations.dtype == bool:

					# Without lambda, iteration of split returns a tuple of
					# value and dataframe
					


					splits = data.groupby(attribute)#.apply(lambda x: x)
					#print(splits)
					GAIN = self.parent_entropy
					SPLIT_INFO = 0
					tmp = []
					for s in splits.groups:
						split = splits.get_group(s)
						#split.pop(attribute), data.pop(attribute)
						#print(split)
						classifications = split[label]
						#print(classifications)
						#print(classifications)
						pi = self.relative_occurrence(classifications)
						#print(pi)
						n_split = len(classifications)
						#print(n_split)
						#print(split)
						GAIN -=  n_split*self.shannon_entropy(pi)/self.N
						SPLIT_INFO -= n_split*np.log2(n_split/self.N)/self.N
						#print(GAIN)
						tmp.append(split)
						#print(GAIN)
					#print(tmp)
					gains[GAIN] = [
						(attribute, True),
						tmp
					]

				else:
					data = data.sort_values(attribute).reset_index(drop=True)
					#print(data)

					for index in data.index:
						
						if (index > 1) and (index < max(data.index)-1):
							less_than, greater_than = data.iloc[:index], data.iloc[index:]
							n_less, n_greater = len(less_than), len(greater_than)

							SPLIT_INFO = -n_greater*np.log2(n_greater/self.N)/self.N - -n_less*np.log2(n_less/self.N)/self.N
							pi_left = self.relative_occurrence(greater_than[label])
							pi_right = self.relative_occurrence(less_than[label])

							GAIN_LEFT = n_greater*self.shannon_entropy(pi_left)/self.N
							GAIN_RIGHT = n_less*self.shannon_entropy(pi_right)/self.N
							#print(GAIN_LEFT, GAIN_RIGHT)
							#print(less_than[label])
							#print(greater_than[label])
							#splits = self.split(data, index)
							#INFO = 0
							#for s in splits:
							#	print(s)
							#	INFO += self.info(s[label])
							GAIN = self.parent_entropy - GAIN_LEFT - GAIN_RIGHT

							#print(GAIN)
							"""
							less_than, greater_than = data.iloc[:index], data.iloc[index:]
							#print(less_than[label])
							#print(greater_than[label])
							#splits = self.split(data, index)
							#INFO = 0
							#for s in splits:
							#	print(s)
							#	INFO += self.info(s[label])
							GAIN = self.parent_entropy - self.info(less_than[label]) - self.info(greater_than[label])
							"""
							# Order of splits is reveresed to have left output
							# always be greater than for numerical attributes
							gains[GAIN] = [
								(attribute, data[attribute][index]),
								[greater_than, less_than]
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
			split_attribute, condition = self.condition
			#split_attribute, condition = self.parent
			
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
				if isinstance(branch, Tree):
					if branch.leaf is not None:
							#print(branch.leaf)
							tmp.append(branch.leaf)
							RUN_PREDICTION = False
					else:
						pass

		predictions = pd.Series(tmp)
		return predictions


# CONTROL VARIABLES/OBJECTS ###################################################
# Bagging params
DATA_FRACTION = 0.2
ENSEMBLE_SIZE = 1
#FOREST_RATE = 1

# Decision tree params
TRAIN_RATIO = 0.8#0.002
depth = 3
save = False
show = False
gain_v_depth = False
rate_v_depth = False


test = False
perf_v_size = False
# Fixed random seed for reproducibility
#np.random.seed(12345)


# LOAD DATA ###################################################################
# Dataframe
#df = pd.read_csv('siren_data_train.csv')


# Variables


#NUM_HEARD = len(df[df['heard'] == 1])
#NUM_NOT_HEARD = NUM_ROWS - NUM_HEARD
#print(NUM_HEARD, NUM_NOT_HEARD)
#df = df.iloc[0:NUM_DATA_POINTS]
#print(df)


# PRE-PROCESSING ##############################################################

# Filter data

#df.pop('near_fid')
#df = df[df['asleep'] == 0] # Filters to non-sleeping instances

# Convert coords to log-distances between individual and fid
#dist_x = df.pop('near_x') - df.pop('xcoor')
#dist_y = df.pop('near_y') - df.pop('ycoor')
#dist = np.sqrt(dist_x**2 + dist_y**2)
#df['log_dist'] = np.log(dist)

# Select columns included in the preamble
#print(df.columns)



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
	'Unnamed: 0':'Categorical', 	# categorical, should this be included?
	'near_angle':'Numerical',	# Uniform on x \in [-180,180]
	'near_x':'Numerical',
	'near_y':'Numerical',
	'xcoor':'Numerical',
	'ycoor':'Numerical',
	'distance':'Numerical',
	'distance_log':'Numerical',
	'sin_angle':'Numerical',
	'cos_angle':'Numerical',
	'age':'Numerical',			# Almost uniform on x \in [20,80]
	'heard':bool,			# label: N/A
	'building':bool,		# bool
	'noise':bool,			# bool
	'in_vehicle':bool,		# bool
	'no_windows':bool,		# bool
	'asleep':bool,			# bool
}
train_df = pd.read_csv('siren_data_train_TRAIN.csv')
test_df = pd.read_csv('siren_data_train_TEST.csv')

NUM_ROWS = len(train_df.index)
NUM_ROWS_USE = int(NUM_ROWS*DATA_FRACTION)
train_df = train_df[:NUM_ROWS_USE]

#print(train_df.columns)
for attribute in train_df.columns:
	#print(attribute)
	if attribute_type[attribute] == bool:
		train_df[attribute] = train_df[attribute].astype(bool)
		test_df[attribute] = test_df[attribute].astype(bool)


#def classification_tree(train_data, test_data, label, max_depth, ensemble_size=1):
def classification_tree(train_df, test_df, label, train_ratio, max_depth, ensemble_size=1):
	pass

	bag = []
	for b in range(ensemble_size):
		#np.random.seed(123)
		tmp_df = copy.deepcopy(train_df)
		tmp_df = tmp_df.sample(frac=1, replace=True).reset_index(drop=True)
		NUM_ROWS = len(tmp_df.index)
		NUM_TRAIN = int(NUM_ROWS*train_ratio)

		# Extract and split features
		#train_df = tmp_df.iloc[:NUM_TRAIN]
		#test_df = tmp_df.iloc[NUM_TRAIN:]


		if ensemble_size == 1:
			print('Fitting...')
		else:
			print(f'Fitting system {b}')

		tree = Tree(train_df, 'heard', max_depth)
		tree.grow()
		
		nodes, leaves = tree.node_count, tree.leaf_count
		print('Nodes: ',nodes)
		print('Leaves: ', leaves)
		
		if nodes > 1:
			predict = tree.predict(test_df, label)
			bag.append(predict)
			#print(predict)
			#del tree

			# Single decision tree using sklearn for comparison
			train = copy.deepcopy(train_df)
			test = copy.deepcopy(test_df)

			clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth, max_features='sqrt')
			y_train = train.pop(label)
			y_test = test.pop(label)
			clf.fit(train, y_train)
			predictions = clf.predict(test)
			#print(predictions)
			#print(clf.tree_.max_depth)
			sklearn_misclass = 1-accuracy_score(y_test, predictions)
			#print(predictions, len(predictions))


		#print(predict)
	
	print('Pooling...')
	bag_df = pd.DataFrame(np.array(bag).T)
	majority_prediction = bag_df.mode(axis=1)[0]
	
	#print(bag_df)
	#print(majority_prediction)

	#print(bag_df)

	print('Evaluating performance...')
	# Confusion matrix
	confusion_matrix = np.zeros((2,2))
	#print(confusion_matrix)

	for i, j in zip(test_df[label], majority_prediction):
		if (i == True) and (j == True):
			confusion_matrix[0][0] += 1
		elif (i == False) and (j == True):
			confusion_matrix[0][1] += 1
		elif (i == True) and (j == False):
			confusion_matrix[1][0] += 1
		elif (i == False) and (j == False):
			confusion_matrix[1][1] += 1
	print(confusion_matrix)
	MISCLASSIFICATION_RATE = (confusion_matrix[0, 1] + confusion_matrix[1, 0])/np.sum(confusion_matrix)
	ACCURACY = 1 - MISCLASSIFICATION_RATE


	print(f'Misclassification rate:\t{MISCLASSIFICATION_RATE:.4f}\t({sklearn_misclass:.4f} using sklearn)')
	print(f'Accuracy:\t\t\t\t{ACCURACY:.4f}')
	return (MISCLASSIFICATION_RATE, ACCURACY)


"""	
#np.random.seed(123)
tmp_df = copy.deepcopy(df)
tmp_df = tmp_df.sample(frac=DATA_FRACTION).reset_index(drop=True)
NUM_ROWS = len(tmp_df.index)
NUM_TRAIN = int(NUM_ROWS*TRAIN_RATIO)

# Extract and split features
train_df = tmp_df.iloc[:NUM_TRAIN]
test_df = tmp_df.iloc[NUM_TRAIN:]
"""

#df = df.sample(frac=DATA_FRACTION).reset_index(drop=True)



if perf_v_size:
	misclass_rates = []
	sizes = np.arange(ENSEMBLE_SIZE)+1
	
	for size in sizes:
		misclass_rate, accuracy = classification_tree(df, 'heard', TRAIN_RATIO, depth, size)
		misclass_rates.append(misclass_rate)
	
	plt.title(f'Effects of ensemble size on performance')
	plt.xlabel('Ensemble size')
	plt.ylabel('Misclassification rate')
	plt.plot(sizes, misclass_rates, '.-')

elif test is False:

	#result, tree = classification_tree(train_df, test_df, 'heard', depth, ENSEMBLE_SIZE)
	misclassification_rate, accuracy = classification_tree(train_df, test_df, 'heard', TRAIN_RATIO, depth, ENSEMBLE_SIZE)


	
	if gain_v_depth:
		x = tree.gains.keys(); y = tree.gains.values()
		yy.append(list(y))
		plt.plot(x,y, '.', label=f'Run {run}')

	else:
		pass
		

print('Finished!')

if gain_v_depth:
	out_df = pd.DataFrame(yy)
	out_df['avg'] = out_df.mean(axis=0)
	plt.plot(out_df.index, out_df['avg'], 'k', label='Average', zorder=0) 


	plt.title(f'Depth optimisation with {TRAIN_RATIO:.0%} of data used for training')
	plt.xlabel('Depth')
	plt.ylabel('Max gain')
	plt.legend()

	if save:
		out_df.to_csv(f'{TRAIN_RATIO*100:.0%}trainsize_{depth}maxdepth_{runs}runs.txt', sep=' ')

if show:
	plt.show()