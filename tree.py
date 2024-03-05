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

# CONTROL VARIABLES/OBJECTS ###################################################
# Bagging/random forest params
DATA_FRACTION = 1	# Only used for testing purposes
ENSEMBLE_SIZE = 1
BAGGING_RATIO = 1
NUM_POINTS = 10
max_features = None#'sqrt' #'sqrt' # sqrt or None

# Decision tree params
depth = 3
save = False
show = True

# Modes of analysis
test = True
feature_selection = False


perf_v_depth = False
perf_v_bag_ratio = False
perf_v_ensemble_size = True
random_search = False

# Fixed random seed for reproducibility
seed = 12345
#np.random.seed(seed)

# Naive model params
naive_attribute = 'distance'
naive_threshold = 6000


# LOAD DATA ###################################################################
train_df = pd.read_csv('siren_data_train_TRAIN.csv')
test_df = pd.read_csv('siren_data_train_TEST.csv')


# PRE-PROCESSING ##############################################################
# Express booleans as {True,False} instead of {0,1}
# Also filter unwanted features away by commenting them away
attribute_type = {
	#'near_fid':'Categorical', 	
	#'Unnamed: 0':'Categorical',
	'near_angle':'Numerical',		# can be removed
	'near_x':'Numerical',			# can be removed
	'near_y':'Numerical',			# can be removed
	'xcoor':'Numerical',			# can be removed
	'ycoor':'Numerical',			# can be removed
	'cos_angle':'Numerical',		# can be removed
	'sin_angle':'Numerical',
	'distance':'Numerical',
	'distance_log':'Numerical',
	'age':'Numerical',
	'noise':bool,
	'no_windows':bool,
	'building':bool,				# can be removed
	'in_vehicle':bool,				# can be removed
	'asleep':bool,					# can be removed
	'heard':bool,	
}

NUM_ROWS = len(train_df.index)
NUM_ROWS_USE = int(NUM_ROWS*DATA_FRACTION)
train_df = train_df[:NUM_ROWS_USE]

# Filter attributes and convert type to bool
for attribute in train_df.columns:
	if attribute in attribute_type.keys():
		if attribute_type[attribute] == bool:
			train_df[attribute] = train_df[attribute].astype(bool)
			test_df[attribute] = test_df[attribute].astype(bool)
	else:
		train_df.pop(attribute)
		test_df.pop(attribute)


# FUNCTIONS ###################################################################
class Node:
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
	Breiman (1984) and Quinlan (1986) that employs unnormalised information
	gain (similar to ID3).
	
	Current implementation does not support regression or more than two classes,
	i.e. not a CART algorithm."""
	features = {}
	gains = {}

	def __init__(self,
		data,
		label,
		MAX_DEPTH=None,
		FOREST_RATE=1,
		NUM_PARENTS=0,
		max_features=None,
		*args
	):
		super().__init__(*args)
		self.X = data
		self.label = label
		self.FOREST_RATE = FOREST_RATE
		self.NUM_PARENTS = NUM_PARENTS
		self.max_features = max_features

		self.NUM_SAMPLES = len(self.X)
		self.attributes = self.X.columns

		# Shannon entropy of entire dataset
		self.pi = self.relative_occurrence(self.X[self.label])
		self.parent_entropy = self.shannon_entropy(self.pi)
		self.N = len(self.X.index)

		# Resets counters only when MAX_DEPTH is assigned
		if MAX_DEPTH is None:
			pass
		else:
			Tree.MAX_DEPTH = MAX_DEPTH
			Tree.features = {attribute:0 for attribute in self.X.columns}

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

		if self.NUM_PARENTS < Tree.MAX_DEPTH:
			test_freq = self.relative_occurrence(self.X[self.label])

			# If node is pure, forms leaf
			if (test_freq[0] == 1) or (self.N < 2):
				CLASSIFICATION = self.classify(labels)
				self.leaf = CLASSIFICATION
				Node.leaf_count += 1

			# If no leaf, node splits
			else:
				try:
					self.condition, split = self.optimise_split(self.X, self.label)
					split_attribute = self.condition[0]
					Tree.features[split_attribute] += 1

					left, right = split

					# Condition fulfilled
					self.left = Tree(
						left,
						self.label,
						NUM_PARENTS=self.NUM_PARENTS+1,
						max_features=self.max_features
					)
					self.left.grow()

					# Condition not fulfilled
					self.right = Tree(
						right,
						self.label,
						NUM_PARENTS=self.NUM_PARENTS+1,
						max_features=self.max_features
					)
					self.right.grow()
					Node.node_count += 1

				# Convert to leaf if split not possible
				except:
					CLASSIFICATION = self.classify(labels)
					self.leaf = CLASSIFICATION
					Node.leaf_count += 1

		# Convert to leaf upon reaching max. depth
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

	def classify(self, labels):
		"""Determine the class of a set of labels based on a majority vote. """
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
		
		# Using sqrt/log and bagging results in a random forest	
		if self.max_features is None:
			pass
		else:
			tmp_attr = list(data.columns)
			tmp_attr.remove(label)
			d = len(data.columns)

			# Label is dropped, random attributes selected
			if self.max_features == 'sqrt':
				k = math.ceil(np.sqrt(d))
			elif self.max_features == 'log':
				k = math.ceil(np.log2(d))
			random_attributes = list(np.random.choice(tmp_attr, k, replace=False))
			random_attributes.append(label)

			# Takes only data from random attributes and label
			data = data[random_attributes]

		# Attempt splits for each attribute and record information gain
		for attribute in data.columns:
			if attribute != label:
				attribute_realisations = data[attribute]

				# Boolean attributes
				if attribute_realisations.dtype == bool:
					splits = data.groupby(attribute)

					GAIN = self.parent_entropy
					tmp = {}
					for s in splits.groups:
						split = splits.get_group(s)

						# Binary variables are dropped on split
						split.pop(attribute)

						classifications = split[label]
						pi = self.relative_occurrence(classifications)
						n_split = len(classifications)
						GAIN -=  n_split*self.shannon_entropy(pi)/self.N
						tmp[s] = split
					
					# Gather attribute, split value and split dataframes
					gains[GAIN] = [
						(attribute, True),
						[tmp[True], tmp[False]]
					]

				# Numerical attributes
				else:
					# Sort numerical values before split
					data = data.sort_values(attribute).reset_index(drop=True)

					for index in data.index:
						if (index > 1) and (index < max(data.index)-1):
							less_than, greater_than = data.iloc[:index], data.iloc[index:]
							n_less, n_greater = len(less_than), len(greater_than)
							pi_left = self.relative_occurrence(greater_than[label])
							pi_right = self.relative_occurrence(less_than[label])
							GAIN_LEFT = n_greater*self.shannon_entropy(pi_left)/self.N
							GAIN_RIGHT = n_less*self.shannon_entropy(pi_right)/self.N
							GAIN = self.parent_entropy - GAIN_LEFT - GAIN_RIGHT

							# Gather attribute, split value and split dataframes
							gains[GAIN] = [
								(attribute, data[attribute][index]),
								[greater_than, less_than]
							]

		# Select split with highest information gain
		MAX_GAIN = max(gains)
		optimal_split = gains[MAX_GAIN]
		return optimal_split
		del gains

	def predict(self, test_data):
		"""Makes a prediciton on test data based on previous training. The label
		specifies which column in the dataframe is used for classification."""
		# Reset index and prepare output object
		test_data = test_data.reset_index(drop=True)
		tmp = []
		tmp = {}

		# One-dimensional classification
		if isinstance(test_data, pd.Series):

			# Classify each sample in the dataset
			for row, s in test_data.items():

				RUN_PREDICTION = True
				count = 0
				while RUN_PREDICTION:

					# Initial node is self
					if count == 0:

						# If current node is not leaf, pass to left/right 
						# depending on current nodes split condition
						prediction = self.leaf
						if prediction is not None:
							tmp[row] = prediction
							RUN_PREDICTION = False
						else:
							# Get attribute and condition
							split_attribute, condition = self.condition

							# Pass through split
							if (type(s) == float) or (type(s) == int):
								if s >= condition:
									branch = self.left
								else:
									branch = self.right
							elif type(s) == bool:
								if s:
									branch = self.left
								else:
									branch = self.right
					else:
						# If current node is not leaf, pass to left/right 
						# depending on current nodes split condition
						prediction = branch.leaf
						if prediction is not None:
							tmp[row] = prediction
							RUN_PREDICTION = False
						else:
							# Get attribute and condition
							split_attribute, condition = branch.condition

							# Pass through split
							if (type(s) == float) or (type(s) == int):
								if s > condition:
									branch = branch.left
								else:
									branch = branch.right
							elif type(s) == bool:
								if s:
									branch = branch.left
								else:
									branch = branch.right
					count += 1

		# Multi-dimensional classification
		elif isinstance(test_data, pd.DataFrame):

			# Classify each sample in the dataset
			for sample in test_data.itertuples():
				row = sample.index
				
				RUN_PREDICTION = True
				count = 0
				while RUN_PREDICTION:

					# Initial node is self
					if count == 0:

						# If current node is not leaf, pass to left/right 
						# depending on current nodes split condition
						prediction = self.leaf
						if prediction is not None:
							tmp[row] = prediction
							RUN_PREDICTION = False
						else:
							# Get attribute and condition
							split_attribute, condition = self.condition
							s = getattr(sample, split_attribute)

							# Pass through split
							if (type(s) == float) or (type(s) == int):
								if s >= condition:
									branch = self.left
								else:
									branch = self.right
							elif type(s) == bool:
								if s:
									branch = self.left
								else:
									branch = self.right
					else:
						# If current node is not leaf, pass to left/right 
						# depending on current nodes split condition
						prediction = branch.leaf
						if prediction is not None:
							tmp[row] = prediction
							RUN_PREDICTION = False
						else:
							# Get attribute and condition
							split_attribute, condition = branch.condition
							s = getattr(sample, split_attribute)

							# Pass through split
							if (type(s) == float) or (type(s) == int):
								if s > condition:
									branch = branch.left
								else:
									branch = branch.right
							elif type(s) == bool:
								if s:
									branch = branch.left
								else:
									branch = branch.right
					count += 1
					

		predictions = pd.Series(tmp)
		return predictions


def evaluate(truth, prediction, print_matrix=False, positive_rate=True):
	"""Evaluates performance in terms of accuracy and positive rates."""
	# Confusion matrix
	confusion_matrix = np.zeros((2,2))
	for i, j in zip(truth, prediction):
		if (i == True) and (j == True):
			confusion_matrix[0][0] += 1
		elif (i == False) and (j == True):
			confusion_matrix[0][1] += 1
		elif (i == True) and (j == False):
			confusion_matrix[1][0] += 1
		elif (i == False) and (j == False):
			confusion_matrix[1][1] += 1

	# Results
	if print_matrix: print(confusion_matrix)
	MISCLASSIFICATION_RATE = (confusion_matrix[0, 1] + \
		confusion_matrix[1, 0])/np.sum(confusion_matrix)
	ACCURACY = 1 - MISCLASSIFICATION_RATE

	TRUE_POSITIVE_RATE = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + \
		confusion_matrix[1, 0])
	FALSE_POSITIVE_RATE = confusion_matrix[1, 0]/(confusion_matrix[0, 0] + \
		confusion_matrix[1, 0])
	
	if positive_rate: return MISCLASSIFICATION_RATE, \
	(TRUE_POSITIVE_RATE, FALSE_POSITIVE_RATE)
	
	else: return MISCLASSIFICATION_RATE

def classification_tree(
		train_df,
		test_df,
		label,
		train_ratio,
		max_depth,
		ensemble_size=1,
		max_features=max_features
	):
	"""Classifier function that enables running over multiple bags."""
	test_bag = []
	train_bag = []
	naive_bag = []
	
	train_df = copy.deepcopy(train_df).sample(frac=1, replace=False).reset_index(drop=True)
	test = copy.deepcopy(test_df)
	for b in range(ensemble_size):
		if ensemble_size == 1:
			print('Fitting...')
		else:
			print(f'Fitting system {b}')
		
		# Sample data
		train = train_df.sample(frac=train_ratio, replace=True).reset_index(drop=True)
		
		# Fitting train
		tree = Tree(
			train,
			label,
			max_depth,
			max_features=max_features
		)
		tree.grow()
		
		# Tree characteristics output
		nodes, leaves = tree.node_count, tree.leaf_count
		print('Nodes: ',nodes)
		print('Leaves: ', leaves)
		
		# Train predict: average over errors
		train_predict = tree.predict(train)
		MISCLASSIFICATION_RATE_TRAIN = evaluate(train[label], train_predict, positive_rate=False)
		train_bag.append(MISCLASSIFICATION_RATE_TRAIN)

		# Test predict: predict using cumulative model
		test_predict = tree.predict(test)
		test_bag.append(test_predict)

		# Fitting
		naive_train = copy.deepcopy(train)
		naive_train = naive_train[[naive_attribute, label]]
		naive_test = copy.deepcopy(test)
		y_naive = naive_test.pop(label)
		naive_test = naive_test[naive_attribute]

		naive_tree = Tree(
			naive_train,
			label,
			max_depth,
			max_features=max_features
		)
		tree.grow()

		naive_predict = tree.predict(naive_test)
		naive_bag.append(naive_predict)

	# Single decision tree using sklearn for comparison
	if max_features is ('sqrt' or None):
		clf = DecisionTreeClassifier(
			criterion='entropy',
			splitter='best',
			max_depth=max_depth,
			max_features=max_features
		)
	else:
		clf = DecisionTreeClassifier(
			criterion='entropy',
			splitter='best',
			max_depth=max_depth,
		)
	y_train = train.pop(label)
	y_test = test.pop(label)
	clf.fit(train, y_train)
	predictions = clf.predict(test)
	sklearn_misclass_test = 1-accuracy_score(y_test, predictions)
	predictions = clf.predict(train)
	sklearn_misclass_train = 1-accuracy_score(y_train, predictions)


	print('Pooling...')
	test_bag_df = pd.DataFrame(np.array(test_bag).T)
	#train_bag_df = pd.DataFrame(np.array(train_bag).T)
	naive_bag_df = pd.DataFrame(np.array(naive_bag).T)
	#majority_prediction_train = train_bag_df.mode(axis=1)[0]
	majority_prediction_test = test_bag_df.mode(axis=1)[0]
	majority_prediction_naive = naive_bag_df.mode(axis=1)[0]


	print('Evaluating performance...')
	
	#Train data
	print(train_bag)
	MISCLASSIFICATION_RATE_TRAIN = np.average(train_bag)
	print('Train:')
	print(f'  Misclassification rate:\t\t{MISCLASSIFICATION_RATE_TRAIN:.4f}\t({sklearn_misclass_train:.4f} using sklearn)\n')

	# Test data
	MISCLASSIFICATION_RATE_TEST, pr_test = evaluate(y_test, majority_prediction_test)
	TPR, FPR = pr_test
	ACCURACY = 1- MISCLASSIFICATION_RATE_TEST
	print('Test:')
	print(f'  Misclassification rate:\t\t{MISCLASSIFICATION_RATE_TEST:.4f}\t({sklearn_misclass_test:.4f} using sklearn)')
	print(f'  Accuracy:\t\t\t\t\t\t{ACCURACY:.4f}')
	print(f'  TPR (FPR):\t\t\t\t\t{TPR:.4f}\t({FPR:.4f})\n')

	# Naive
	MISCLASSIFICATION_RATE_NAIVE, pr_naive = evaluate(y_naive, majority_prediction_naive)
	TPR, FPR = pr_naive
	print('Naive:')
	print(f'  Misclassification rate:\t\t{MISCLASSIFICATION_RATE_NAIVE:.4f}')
	print(f'  TPR (FPR):\t\t\t\t\t{TPR:.4f}\t({FPR:.4f})')
	
	return (MISCLASSIFICATION_RATE_TRAIN, MISCLASSIFICATION_RATE_TEST, \
		MISCLASSIFICATION_RATE_NAIVE), (pr_test, pr_naive), tree


# GENERATING RESULTS ##########################################################
fig = plt.figure(figsize=(4.5,4.5))

# Simple test run wihtout analysis
if test:
	misclassification_rate, accuracy, tree = classification_tree(
		train_df,
		test_df, 
		'heard', 
		BAGGING_RATIO, 
		depth, 
		ENSEMBLE_SIZE, 
		max_features=max_features
	)

elif feature_selection:
	tmp = []
	runs = 10
	for _ in range(runs):
		misclassification_rate, accuracy, tree = classification_tree(
			train_df,
			test_df, 
			'heard', 
			BAGGING_RATIO, 
			depth, 
			ENSEMBLE_SIZE, 
			max_features=max_features
		)
		tmp.append(tree.features)
	
	features = {k:v for x in tmp for k,v in x.items()}
	plt.bar(features.keys(), features.values())
	
	plt.title(f'Barplot showing feature frequency from {runs} runs using a depth of {depth}')
	plt.xticks(rotation=-45)
	plt.tight_layout()
	plt.show()


else:
	# Random search functionality for two parameters.
	if random_search:
		print('Performing a random search...')
		depths = np.random.randint(1, depth, NUM_POINTS)
		sizes = np.random.randint(2, ENSEMBLE_SIZE, NUM_POINTS)
		
		data = {'depth':depths, 'ensemble size':sizes}
		df = pd.DataFrame(data)

		# Random search
		for d, s in zip(depths, sizes):
			misclass_rates, p_rates, tree = classification_tree(
				train_df,
				test_df, 
				'heard', 
				BAGGING_RATIO, 
				d, 
				s, 
				max_features=max_features
			)
			train_rate, test_rate, naive_rate = misclass_rates
			test_pr, naive_pr = p_rates
			test[s] = [test_rate, test_pr]
		
		# Results
		test_rates = [result[0] for result in test.values()]
		df['test error'] = test_rates
		
		# Get params where test error is lowest
		optimal_params = df[df['test error']==df['test error'].min()]

		print('Random search:')
		print(df)
		print('\nOptimal settings:')
		print(optimal_params)

	# Performance dependence on ensemble size
	elif perf_v_ensemble_size:
		
		# Evaluate per size
		sizes = np.arange(ENSEMBLE_SIZE)+1

		if seed is None:
			seed = np.random.randint(1e5)
			np.random.seed(seed)
			print(seed)
		else:
			pass
			np.random.randint(seed)

		train, test, naive = {}, {}, {}
		for size in sizes:
			print(f'EVALUATING ENSEMBLE SIZE: {size}')
			# Setting the same seed each time allows the fitting to reach the same
			# points when re-run for an increasing maximum depth
			np.random.seed(seed)

			# Fitting and evaluating performance
			misclass_rates, p_rates, tree = classification_tree(
						train_df,
						test_df, 
						'heard', 
						BAGGING_RATIO, 
						depth, 
						size, 
						max_features=max_features
			)
			train_rate, test_rate, naive_rate = misclass_rates
			test_pr, naive_pr = p_rates

			train[size] = train_rate
			test[size] = [test_rate, test_pr]
			naive[size] = [naive_rate, naive_pr]
			
		
		test_rates = [result[0] for result in test.values()]
		naive_rates = [result[0] for result in naive.values()]

		plt.plot(train.keys(), train.values(), label='Train error')
		plt.plot(test.keys(), test_rates, label='Test error')
		plt.plot(naive.keys(), naive_rates, label='Naive test error') 

		if max_features == 'sqrt':
			plt.title(f'Training error for increasing an ensemble size of a\nrandom forest using a maximum depth of {depth}')
		else:
			plt.title(f'Training error for increasing an ensemble size of a\nrandom tree using a maximum depth of {depth}')
		
		plt.xlabel('Ensemble size')
		plt.ylabel('Misclassification rate')
		

	elif perf_v_bag_ratio:
		# Evaluate per bag ratio
		ratios = np.round(np.linspace(0.1, 1, NUM_POINTS), 2)
		
		if seed is None:
			seed = np.random.randint(1e5)
			np.random.seed(seed)
			print(seed)
		else:
			np.random.randint(seed)

		train, test, naive = {}, {}, {}
		for ratio in ratios:
			print(f'EVALUATING RATIO: {ratio}')
			# Setting the same seed each time allows the fitting to reach the 
			# same points when re-run for an increasing maximum depth
			np.random.seed(seed)

			# Fitting and evaluating performance
			misclass_rates, p_rates, tree = classification_tree(
						train_df,
						test_df, 
						'heard', 
						ratio, 
						depth, 
						ENSEMBLE_SIZE, 
						max_features=max_features
			)
			train_rate, test_rate, naive_rate = misclass_rates
			test_pr, naive_pr = p_rates

			train[ratio] = train_rate
			test[ratio] = [test_rate, test_pr]
			naive[ratio] = [naive_rate, naive_pr]
			
		
		test_rates = [result[0] for result in test.values()]
		naive_rates = [result[0] for result in naive.values()]

		plt.plot(train.keys(), train.values(), label='Train error')
		plt.plot(test.keys(), test_rates, label='Test error')
		plt.plot(naive.keys(), naive_rates, label='Naive test error') 

		plt.title(f'Training error for increasing bag ratio using a\nmaximum depth of {depth}')
		if ENSEMBLE_SIZE > 1:
			plt.title(f'Training error for increasing bag ratio using a\nmaximum depth of {depth} and an ensemble size of {ENSEMBLE_SIZE}')
		
		plt.xlabel('Bag ratio')
		plt.ylabel('Misclassification rate')


	# Training error evaluation
	elif perf_v_depth:
		if seed is None:
			seed = np.random.randint(1e5)
			np.random.seed(seed)
			print(seed)
		else:
			np.random.randint(seed)

		train, test, naive = {}, {}, {}
		for d in range(1,depth):
			print(f'EVALUATING DEPTH: {d}')
			# Setting the same seed each time allows the fitting to reach the 
			# same points when re-run for an increasing maximum depth
			np.random.seed(seed)

			# Fitting and evaluating performance
			misclass_rates, p_rates, tree = classification_tree(
						train_df,
						test_df, 
						'heard', 
						BAGGING_RATIO, 
						d, 
						ENSEMBLE_SIZE, 
						max_features=max_features
			)
			train_rate, test_rate, naive_rate = misclass_rates
			test_pr, naive_pr = p_rates

			train[d] = train_rate
			test[d] = [test_rate, test_pr]
			naive[d] = [naive_rate, naive_pr]
			
		
		test_rates = [result[0] for result in test.values()]
		naive_rates = [result[0] for result in naive.values()]

		plt.plot(train.keys(), train.values(), label='Train error')
		plt.plot(test.keys(), test_rates, label='Test error')
		plt.plot(naive.keys(), naive_rates, label='Naive test error') 

		plt.title(f'Training error for increasing depth')
		if ENSEMBLE_SIZE > 1:
			if max_features == 'sqrt':
				plt.title(f'Training error for increasing depth using \nan ensemble size of {ENSEMBLE_SIZE} in a random forest')
			else:
				plt.title(f'Training error for increasing depth using \nan ensemble size of {ENSEMBLE_SIZE}')

		plt.xlabel('Maximum depth')
		plt.ylabel('Misclassification rate')

		if save:
			out_df.to_csv(f'{TRAIN_RATIO*100:.0%}trainsize_{depth}maxdepth_{runs}runs.txt', sep=' ')
	

	print('Finished!')
	if show:
		plt.tight_layout()
		plt.legend()
		plt.show()