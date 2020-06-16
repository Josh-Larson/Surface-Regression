import numpy as np
import features
import random as rand
import math as m


class ModelUpdates:
	def __init__(self, line_count=1):
		self.line_count = line_count
		self.updates = np.zeros((line_count, 15), dtype=np.float64)
		self.update_count = np.zeros((line_count, 1), dtype=np.int32)
	
	def copy(self):
		return ModelUpdates(self.line_count)
	
	def clear(self):
		self.updates[:, :] = 0
		self.update_count[:, :] = 0
		
	def adjust_by_count(self):
		self.updates /= self.update_count
	
	def add_line(self):
		self.updates = np.concatenate((self.updates, np.zeros((1, 15), dtype=np.float64)))
		self.update_count = np.concatenate((self.update_count, np.zeros((1, 1), dtype=np.float64)))
		self.line_count += 1
	
	def get_unused_lines(self):
		return (self.update_count == 0).flatten()
	
	def remove_lines(self, lines):
		assert(len(lines) == self.line_count)
		self.updates = self.updates[~lines]
		self.update_count = self.update_count[~lines]
		self.line_count = len(self.update_count)


class Model:
	def __init__(self, points):
		self.points = points
		self.lines = np.zeros((1, 22), dtype=np.float64)
		self.nearby_line = np.zeros((len(points), 8), dtype=np.float64)
		self.updates = ModelUpdates()
		
		# Random initial line
		self.lines[0, 0:3] = points[rand.randint(0, len(points)-1)]
		self.lines[0, 3:6] = points[rand.randint(0, len(points)-1)] - self.lines[0, 0:3]
		self.normalize_lines()
		features.update_orthogonalization(self.lines)
	
	def copy(self):
		ret = Model(self.points)
		ret.lines = self.lines.copy()
		ret.nearby_line = self.nearby_line.copy()
		ret.updates = self.updates.copy()
		return ret
	
	def get_rmse_linear(self):
		return m.sqrt(np.mean(self.nearby_line[:, 2]))
	
	def get_rmse_topology(self, line):
		return m.sqrt(np.mean(self.nearby_line[:, 2]))
		# if line < 0:
		# 	return np.mean(self.updates.updates[:, 14])
		# else:
		# 	return m.sqrt(abs(self.updates.updates[line, 14]))
	
	def sanity_check(self):
		assert not np.isnan(self.lines).any(), "NaN: lines"
		assert not np.isnan(self.nearby_line).any(), "NaN: nearby_line"
		assert not np.isnan(self.updates.updates).any(), "NaN: updates"
		assert not np.isnan(self.updates.update_count).any(), "NaN: update_count"
		
		assert not np.isinf(self.lines).any(), "Inf: lines"
		assert not np.isinf(self.nearby_line).any(), "Inf: nearby_line"
		assert not np.isinf(self.updates.updates).any(), "Inf: updates"
		assert not np.isinf(self.updates.update_count).any(), "Inf: update_count"
	
	def get_unused_lines(self):
		return self.updates.get_unused_lines()

	def remove_unused_lines(self):
		unused_lines = self.updates.get_unused_lines()
		self.lines = self.lines[~unused_lines]
		self.updates.remove_lines(unused_lines)
		return unused_lines
	
	def add_line(self, line):
		self.lines = np.concatenate((self.lines, line.reshape((1, 22))))
		self.normalize_lines()
		self.updates.add_line()
	
	def normalize_lines(self):
		self.sanity_check()
		# for line in range(len(self.lines)):
		# 	line_length = self.lines[line, 3:6]
		# 	magnitude = np.sum(line_length * line_length)
		# 	assert(magnitude > 0)
		# 	self.lines[line, 6:9] = line_length / magnitude

	def initialize_linear(self):
		self.sanity_check()
		# features.initialize(self.points, self.lines, self.nearby_line)
		# nan_view = np.isnan(self.lines).any(axis=1)
		# self.lines = self.lines[~nan_view]
