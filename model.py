from numba import cuda
import numpy as np
import features
import random as rand
import math as m
import time


class Model:
	def __init__(self, points):
		self.points = points
		self.lines = np.zeros((1, 9), dtype=np.float32)
		self.topologies = np.zeros((1, 13), dtype=np.float32)
		self.nearby_line = np.zeros((len(points), 8), dtype=np.float64)
		self.polar_updates = np.zeros((1, 8), dtype=np.float64)
		self.linear_updates = np.zeros((1, 7), dtype=np.float64)
		self.update_count = np.zeros((1, 1), dtype=np.int32)
		
		# State Storage
		self.lines_copy = self.lines.copy()
		self.topologies_copy = self.topologies.copy()
		
		# Random initial line
		self.lines[0, 0:3] = points[rand.randint(0, len(points)-1)]
		self.lines[0, 3:6] = points[rand.randint(0, len(points)-1)] - self.lines[0, 0:3]
		self.normalize_lines()
	
	def sanity_check(self):
		if np.isnan(self.lines).any():
			print("NaN: lines")
			assert False
		if np.isnan(self.topologies).any():
			print("NaN: topologies")
			assert False
		if np.isnan(self.nearby_line).any():
			print("NaN: nearby_line")
			assert False
		if np.isnan(self.polar_updates).any():
			print("NaN: polar_updates")
			assert False
		if np.isnan(self.linear_updates).any():
			print("NaN: linear_updates")
			assert False
		if np.isnan(self.update_count).any():
			print("NaN: update_count")
			assert False
		if np.isinf(self.lines).any():
			print("Inf: lines")
			assert False
		if np.isinf(self.topologies).any():
			print("Inf: topologies")
			assert False
		if np.isinf(self.nearby_line).any():
			print("Inf: nearby_line")
			assert False
		if np.isinf(self.polar_updates).any():
			print("Inf: polar_updates")
			assert False
		if np.isinf(self.linear_updates).any():
			print("Inf: linear_updates")
			assert False
		if np.isinf(self.update_count).any():
			print("Inf: update_count")
			assert False

	def normalize_lines(self):
		self.sanity_check()
		for line in range(len(self.lines)):
			line_length = self.lines[line, 3:6]
			magnitude = np.sum(line_length * line_length)
			if magnitude > 0:
				self.lines[line, 6:9] = line_length / magnitude
			else:
				self.lines[line, 6:9] = np.nan

	def initialize_linear(self):
		self.sanity_check()
		# features.initialize_linear(self.points, self.lines, self.topologies, self.nearby_line)
		features.get_nearest_neighbors(self.points, self.lines, self.nearby_line)
		features.extend_line_segments(self.nearby_line, self.lines)
		features.update_orthogonalization(self.lines, self.topologies)
		nan_view = np.isnan(self.lines).any(axis=1)
		self.lines = self.lines[~nan_view]
		self.topologies = self.topologies[~nan_view]
	
	def get_rmse_polar(self):
		self.sanity_check()
		return m.sqrt(np.mean(self.polar_updates[:, 7]))
	
	def get_rmse_linear(self):
		self.sanity_check()
		return m.sqrt(np.mean(self.nearby_line[:, 2]))
	
	def state_store(self):
		self.sanity_check()
		self.lines_copy = self.lines.copy()
		self.topologies_copy = self.topologies.copy()
		return self.lines_copy.copy(), self.topologies_copy.copy()
	
	def state_restore(self):
		self.sanity_check()
		lines_copy = self.lines.copy()
		topologies_copy = self.topologies.copy()
		self.lines = self.lines_copy
		self.topologies = self.topologies_copy
		return lines_copy, topologies_copy

	def calculate_polar_updates(self):
		self.sanity_check()
		self.polar_updates[:, :] = 0
		self.update_count[:, :] = 0
		features.calculate_topology_updates(self.topologies, self.nearby_line, self.polar_updates, self.update_count)
		# self.update_count[np.where(self.update_count == 0)] = 1
		# nan_view = np.isnan(self.update_count).any(axis=1)
		# self.polar_updates = self.polar_updates[~nan_view]
		# self.update_count = self.update_count[~nan_view]
		update_removal = (self.update_count == 0).flatten()
		self.polar_updates = self.polar_updates[~update_removal]
		self.update_count = self.update_count[~update_removal]
		self.polar_updates /= self.update_count
		return not np.isnan(self.polar_updates).any() and not np.isinf(self.polar_updates).any()
	
	def calculate_linear_updates(self):
		self.sanity_check()
		self.linear_updates[:, :] = 0
		self.update_count[:, :] = 0
		features.calculate_linear_updates(self.nearby_line, self.linear_updates, self.update_count)
		self.linear_updates /= self.update_count
		return not np.isnan(self.linear_updates).any() and not np.isinf(self.linear_updates).any()
	
	def polar_regression_test(self, line, alpha):
		self.sanity_check()
		lines = self.lines.copy()
		topologies = self.topologies.copy()
		if not self.calculate_polar_updates():
			self.topologies[:, :] = topologies
			return False, 0.
		update1 = self.get_rmse_polar()
		update2 = update1
		for i in range(4):
			self.topologies[:, :7] += alpha * self.polar_updates[line, :7]
			if not self.calculate_polar_updates():
				self.topologies[:, :] = topologies
				return False, 0.
			update2 = self.get_rmse_polar()
			if update2 > update1:
				return False, 0.
		self.topologies = topologies
		self.lines = lines
		return True, update1 - update2

	def polar_regression_find_best(self, line):
		self.sanity_check()
		learning_rate = 1e-5
		amount = 0.
		for learning_rate_exp in range(-1, -6, -1):
			alpha = pow(10, learning_rate_exp)
			for i in range(2):
				alpha_iteration = alpha * 3 if i == 0 else alpha
				converged, test = self.polar_regression_test(line, alpha_iteration)
				if converged and test > amount:
					learning_rate = alpha_iteration
					amount = test
				elif amount > 0:
					break
		return learning_rate

	def polar_regression(self, update_threshold=1e-5, max_iterations=np.inf):
		self.sanity_check()
		previous_update = np.inf
		update_difference = update_threshold + 1
		learning_rate = np.zeros((len(self.polar_updates), 1))
		for l in range(len(self.polar_updates)):
			learning_rate[l] = self.polar_regression_find_best(l)
		iteration = 0
		
		while update_difference >= update_threshold and iteration < max_iterations:
			if not self.calculate_polar_updates():
				return False
			self.topologies[:, :7] += learning_rate * self.polar_updates[:, :7]
			largest_update = self.get_rmse_polar()
			update_difference = previous_update - largest_update
			previous_update = largest_update
			iteration += 1
		return True
	
	def linear_regression_test(self, alpha):
		self.sanity_check()
		line_copy = self.lines.copy()
		topologies = self.topologies.copy()
		self.initialize_linear()
		if not self.calculate_linear_updates():
			self.lines[:, :] = line_copy
			return False, 0.
		error1 = self.get_rmse_linear()
		error2 = error1
		for i in range(20):
			self.lines[:, 0:6] += alpha * self.linear_updates[:, 0:6]
			self.initialize_linear()
			if not self.calculate_linear_updates():
				self.lines[:, :] = line_copy
				return False, 0.
			error2 = self.get_rmse_linear()
			if error2 > error1:
				return False, 0.
		self.lines = line_copy
		self.topologies = topologies
		return True, error1 - error2
	
	def linear_regression_find_best(self):
		self.sanity_check()
		learning_rate = 1e-5
		amount = 0.
		for learning_rate_exp in range(-1, -6, -1):
			alpha = pow(10, learning_rate_exp)
			for i in range(2):
				alpha_iteration = alpha * 3 if i == 0 else alpha
				converged, test = self.linear_regression_test(alpha_iteration)
				if converged and test > amount:
					learning_rate = alpha_iteration
					amount = test
				elif amount > 0:
					break
		return learning_rate
	
	def linear_regression(self, update_threshold=1e-7):
		self.sanity_check()
		alpha = self.linear_regression_find_best()
		previous_update = np.inf
		update_difference = update_threshold + 1
		iteration = 0
		print("Linear Regression LR: %f" % alpha)
		while update_difference >= update_threshold:
			if iteration > 0:
				self.lines[:, 0:6] += alpha * self.linear_updates[:, 0:6]
				self.normalize_lines()
			self.initialize_linear()
			if not self.calculate_linear_updates():
				break
			current_update = self.get_rmse_linear()
			update_difference = previous_update - current_update
			previous_update = current_update
			iteration += 1
		self.initialize_linear()
		self.topologies[:, :] = 0.
		self.polar_regression(update_threshold=update_threshold)
		return self.get_rmse_polar(), self.get_rmse_linear(), iteration, update_difference
	
	def augment_lines(self):
		# Precondition: linear_regression has run prior to this
		# Nearest Line: line_idx, intercept, distance, angle, distance_error, dx, dy, dz
		augment_index = int(np.argmax(np.abs(self.nearby_line[:, 2])))
		augment_point = self.nearby_line[augment_index]
		line_augmented = self.lines[int(augment_point[0])]
		intercept = line_augmented[0:3] + line_augmented[3:6] * augment_point[1]
		
		self.initialize_linear()
		
		# Pivot
		self.state_store()
		line_added = np.zeros((9,))
		line_added[0:3] = intercept + augment_point[5:8]
		line_added[3:6] = (line_augmented[0:3] + line_augmented[3:6]) - line_added[0:3]
		self.lines[-1, 3:6] = line_added[0:3] - self.lines[-1, 0:3]
		self.lines = np.concatenate((self.lines, line_added.reshape((1, 9))))
		self.topologies = np.concatenate((self.topologies, np.zeros((1, 13))))
		self.normalize_lines()
		rmse_polar_pivot, rmse_linear_pivot, iterations_pivot, update_difference_pivot = self.linear_regression()
		line_pivot, topology_pivot = self.state_restore()
		
		self.initialize_linear()
		
		# Intercept
		self.state_store()
		length = augment_point[5:8]
		augmented_line = np.zeros((9,))
		augmented_line[0:3] = intercept
		augmented_line[3:6] = length
		augmented_line[6:9] = augmented_line[3:6] / pow(np.linalg.norm(augmented_line[3:6]), 2)
		self.lines = np.concatenate((self.lines, augmented_line.reshape((1, 9))))
		self.topologies = np.concatenate((self.topologies, np.zeros((1, 13))))
		self.normalize_lines()
		rmse_polar_intercept, rmse_linear_intercept, interations_intercept, update_difference_intercept = self.linear_regression()
		line_intercept, topology_intercept = self.state_restore()
		
		if rmse_polar_pivot >= rmse_polar_intercept:
			self.lines = line_pivot
			self.topologies = topology_pivot
			return rmse_polar_pivot, rmse_linear_pivot, iterations_pivot, update_difference_pivot
		else:
			self.lines = line_intercept
			self.topologies = topology_intercept
			return rmse_polar_intercept, rmse_linear_intercept, interations_intercept, update_difference_intercept
