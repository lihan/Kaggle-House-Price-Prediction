from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from source.data_processing import *
import pandas as pd
import numpy as np


class DataSet:
	def __init__(self, data_set):
		self.indexed_data_set = data_set
		self.scaled_data_set = None

		self.scaler = None

		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

		self.setup_data_set()
		self.split_data()

		print("Data Scaled And Separated!")

	def setup_data_set(self):
		"""Sets up the passed Housing Data Set
		1) Removes Null Values
		2) Encodes Data
		3) Scales Data

		Each function is taken from data_processing module
		"""
		null_removed_data = remove_null_values(self.indexed_data_set)
		encoded_data = encode_data(null_removed_data)

		encoded_data = pd.get_dummies(encoded_data).copy()
		self.indexed_data_set = encoded_data

		scaled_data_results = scale_data(encoded_data)
		self.scaled_data_set = scaled_data_results[1]
		self.scaler = scaled_data_results[0]

	def split_data(self):
		X = self.indexed_data_set.drop("SalePrice", axis=1)
		y = self.indexed_data_set['SalePrice']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=.8)


class GradientBoostingMachine(GradientBoostingRegressor, DataSet):
	"""Implementation of SKlearn's GradientBoostingRegressor"""

	def __init__(self, data_set, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
	             criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	             max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None,
	             max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'):
		"""Simply Sets up the Gradient Boosting Regressor init function
		:param data_set: The Pandas DataFrame to be used
		"""
		GradientBoostingRegressor.__init__(self, loss, learning_rate, n_estimators, subsample, criterion,
		                                   min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
		                                   max_depth, min_impurity_decrease, min_impurity_split, init,
		                                   random_state, max_features, alpha, verbose, max_leaf_nodes,
		                                   warm_start, presort)
		DataSet.__init__(self, data_set)

	def _make_estimator(self, append=True, random_state=None):
		"""we don't need _make_estimator"""
		raise NotImplementedError()

	def validate_model(self): pass


class LassoRegressor(Lasso, DataSet):
	def __init__(self, data_set, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True,
	             max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		Lasso.__init__(self, alpha, fit_intercept, normalize, precompute, copy_X, max_iter,
		               tol, warm_start, positive, random_state, selection)
		DataSet.__init__(self, data_set)


data = pd.read_csv('../input/train.csv')
test = GradientBoostingMachine(data, n_estimators=100000)
test.fit(test.X_train, test.y_train)
print(test.score(test.X_train, test.y_train))
print(test.score(test.X_test, test.y_test))


other_test = LassoRegressor(data)
other_test.fit(other_test.X_train, other_test.y_train)
print(other_test.score(other_test.X_train, other_test.y_train))
print(other_test.score(other_test.X_test, other_test.y_test))
