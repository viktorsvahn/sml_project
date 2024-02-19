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
# Bagging params
DATA_FRACTION = 1
ENSEMBLE_SIZE = 2
BAGGING_RATIO = 0.2

max_features = None#'sqrt' # sqrt or None

# Decision tree params
depth = 2
save = False
show = True



test = False
gain_v_depth = False

ROC = True
perf_v_size = False
perf_v_bag_ratio = False
run_multiplicity = 1
# Fixed random seed for reproducibility
#np.random.seed(123456)

classifiers = []
results = []




# LOAD DATA ###################################################################
train_df = pd.read_csv('siren_data_train_TRAIN.csv')
test_df = pd.read_csv('siren_data_train_TEST.csv')


# PRE-PROCESSING ##############################################################
train_df.pop('near_fid')
train_df.pop('Unnamed: 0')
test_df.pop('near_fid')
test_df.pop('Unnamed: 0')

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

NUM_ROWS = len(train_df.index)
NUM_ROWS_USE = int(NUM_ROWS*DATA_FRACTION)
train_df = train_df[:NUM_ROWS_USE]

#print(train_df.columns)
for attribute in train_df.columns:
	#print(attribute)
	if attribute_type[attribute] == bool:
		train_df[attribute] = train_df[attribute].astype(bool)
		test_df[attribute] = test_df[attribute].astype(bool)


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

			print(self.NUM_SAMPLES)
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

			# If no leaf, trey split
			else:
				try:
					self.condition, split = self.optimise_split(self.X, self.label)
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
			print('lol')
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
		#print(class_counts)
		
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
		
		#if self.max_features == 'sqrt':
		#	#print('lol')
		#	tmp_attr = list(data.columns)
		#	#print(tmp_attr)
		#	tmp_attr.remove(label)
		#	#print(tmp_attr)
		#	d = len(data.columns)
		#	k = math.ceil(np.sqrt(d))
		#	random_attributes = list(np.random.choice(tmp_attr, k, replace=False))
		#	random_attributes.append(label)
		#	#print(random_attributes)
		#	data = data[random_attributes]
		#else:
		#	pass
		#print(data.columns)
		#print(self.max_features)
		if self.max_features == 'sqrt':
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
		else:
			pass

		for attribute in data.columns:
			if attribute != label:
				attribute_realisations = data[attribute]
				#print(attribute_realisations)
				if attribute_realisations.dtype == bool:

					splits = data.groupby(attribute)
					GAIN = self.parent_entropy
					tmp = {}
					for s in splits.groups:
						split = splits.get_group(s)
						classifications = split[label]
						pi = self.relative_occurrence(classifications)
						n_split = len(classifications)
						GAIN -=  n_split*self.shannon_entropy(pi)/self.N
						tmp[s] = split

					gains[GAIN] = [
						(attribute, True),
						[tmp[True], tmp[False]]
					]

				else:
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

							gains[GAIN] = [
								(attribute, data[attribute][index]),
								[greater_than, less_than]
							]

		MAX_GAIN = max(gains)
		optimal_split = gains[MAX_GAIN]
		return optimal_split
		del gains

	def predict(self, test_data, label):
		"""Makes a prediciton on test data based on previous training. The label
		specifies which column in the dataframe is used for classification."""
		# Reset index and prepare output object
		test_data = test_data.reset_index(drop=True)
		tmp = []
		tmp = {}

		# Classify each sample in the dataset
		for _ in test_data.iterrows():
			row, sample = _
			
			#branch = None
			
			RUN_PREDICTION = True
			count = 0
			while RUN_PREDICTION:
				if count == 0:
					prediction = self.leaf
					if prediction is not None:
						tmp[row] = prediction
						RUN_PREDICTION = False
					else:
						split_attribute, condition = self.condition
						s = sample[split_attribute]
						if (type(s) == float) or (type(s) == int):
							if s >= condition:
								#branch = copy.deepcopy(self.left)
								branch = self.left
							else:
								#branch = copy.deepcopy(self.right)
								branch = self.right
						elif type(s) == bool:
							if s:
								#branch = copy.deepcopy(self.left)
								branch = self.left
							else:
								#branch = copy.deepcopy(self.right)
								branch = self.right


				else:
					prediction = branch.leaf
					if prediction is not None:
						tmp[row] = prediction
						RUN_PREDICTION = False
					else:
						split_attribute, condition = branch.condition
						s = sample[split_attribute]

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


def classification_tree(train_df, test_df, label, train_ratio, max_depth, ensemble_size=1, max_features=max_features):
	"""Classifier function that enables running over multiple bags
	"""
	bag = []
	for b in range(ensemble_size):
		#np.random.seed(123)
		train = copy.deepcopy(train_df)
		train = train.sample(frac=train_ratio, replace=True).reset_index(drop=True)
		test = copy.deepcopy(test_df)
		if ensemble_size == 1:
			print('Fitting...')
		else:
			print(f'Fitting system {b}')

		tree = Tree(train, 'heard', max_depth, max_features=max_features)
		tree.grow()
		
		nodes, leaves = tree.node_count, tree.leaf_count
		print('Nodes: ',nodes)
		print('Leaves: ', leaves)
		
		if nodes > 0:
			
			predict = tree.predict(test, label)
			bag.append(predict)
			

			# Single decision tree using sklearn for comparison
			clf = DecisionTreeClassifier(
				criterion='entropy',
				splitter='best',
				max_depth=max_depth,
				max_features=max_features
			)
			y_train = train.pop(label)
			y_test = test.pop(label)
			clf.fit(train, y_train)
			predictions = clf.predict(test)
			sklearn_misclass = 1-accuracy_score(y_test, predictions)
			
			#tree.plot_tree(clf)
			#plt.savefig('C:/Users/viksv814/Documents/courses/sml/project/Figure_1.pdf', format='pdf')
			
			print('Pooling...')
			bag_df = pd.DataFrame(np.array(bag).T)
			majority_prediction = bag_df.mode(axis=1)[0]
			
			print('Evaluating performance...')
			
			# Confusion matrix
			confusion_matrix = np.zeros((2,2))
			for i, j in zip(test_df[label], majority_prediction):
				if (i == True) and (j == True):
					confusion_matrix[0][0] += 1
				elif (i == False) and (j == True):
					confusion_matrix[0][1] += 1
				elif (i == True) and (j == False):
					confusion_matrix[1][0] += 1
				elif (i == False) and (j == False):
					confusion_matrix[1][1] += 1
			
			# Results
			print(confusion_matrix)
			MISCLASSIFICATION_RATE = (confusion_matrix[0, 1] + confusion_matrix[1, 0])/np.sum(confusion_matrix)
			ACCURACY = 1 - MISCLASSIFICATION_RATE
			TRUE_POSITIVE_RATE = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])
			FALSE_POSITIVE_RATE = confusion_matrix[1, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])

			import sklearn
			sklearn.metrics.roc_curve(test, majority_prediction)

			print(f'Misclassification rate:\t{MISCLASSIFICATION_RATE:.4f}\t({sklearn_misclass:.4f} using sklearn)')
			print(f'Accuracy:\t\t\t\t{ACCURACY:.4f}')
			print(f'True positive rate:\t\t\t{TRUE_POSITIVE_RATE:.4f}')
			print(f'False positive rate:\t\t{FALSE_POSITIVE_RATE:.4f}')
			return (MISCLASSIFICATION_RATE, ACCURACY)


# GENERATING RESULTS ##########################################################
# Performance dependence on ensemble size
if perf_v_size:
	# Evaluate per size
	sizes = np.arange(ENSEMBLE_SIZE)+1
	runs = []
	for size in sizes:
		misclass_rates = []

		# Mulitplie runs for statistics
		for run in range(run_multiplicity):
			misclass_rate, accuracy = classification_tree(
				train_df,
				test_df, 
				'heard', 
				BAGGING_RATIO, 
				depth, 
				size, 
				max_features=max_features
			)
			misclass_rates.append(misclass_rate)
		runs.append(misclass_rates)
	arr = np.array(runs)

	# Plot
	plt.title(f'Effects of ensemble size on performance\nusing max. depth of {depth}')
	plt.xlabel('Ensemble size')
	plt.ylabel('Misclassification rate')
	plt.boxplot(arr.T)

elif perf_v_bag_ratio:
	# Evaluate per bag ratio
	ratios = np.round(np.linspace(0.1, 1, 10), 2)
	runs = []
	for ratio in ratios:
		misclass_rates = []

		# Mulitplie runs for statistics
		for run in range(run_multiplicity):
			misclass_rate, accuracy = classification_tree(
				train_df,
				test_df, 
				'heard', 
				ratio, 
				depth, 
				ENSEMBLE_SIZE, 
				max_features=max_features
			)
			misclass_rates.append(misclass_rate)
		runs.append(misclass_rates)
	arr = np.array(runs)

	# Plot
	plt.title(f'Effects of bagging ratio on performance\nusing max. depth of {depth} and an ensemble size of {ENSEMBLE_SIZE}')
	plt.xlabel('Bagging ratio')
	plt.ylabel('Misclassification rate')
	plt.boxplot(arr.T, labels=ratios)

elif test is False:

	#result, tree = classification_tree(train_df, test_df, 'heard', depth, ENSEMBLE_SIZE)
	misclassification_rate, accuracy = classification_tree(
		train_df,
		test_df, 
		'heard', 
		BAGGING_RATIO, 
		depth, 
		ENSEMBLE_SIZE, 
		max_features=max_features
	)


# Training error evaluation
if gain_v_depth:
	x = tree.gains.keys(); y = tree.gains.values()
	yy.append(list(y))
	plt.plot(x,y, '.', label=f'Run {run}')
	out_df = pd.DataFrame(yy)
	out_df['avg'] = out_df.mean(axis=0)
	plt.plot(out_df.index, out_df['avg'], 'k', label='Average', zorder=0) 


	plt.title(f'Depth optimisation with {TRAIN_RATIO:.0%} of data used for training')
	plt.xlabel('Depth')
	plt.ylabel('Max gain')
	plt.legend()

	if save:
		out_df.to_csv(f'{TRAIN_RATIO*100:.0%}trainsize_{depth}maxdepth_{runs}runs.txt', sep=' ')


print('Finished!')
if show:
	plt.show()