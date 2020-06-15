from numba import cuda, njit, prange
import numpy as np
import features_cuda as features


@njit
def extend_line_segments(points, lines):
	min_intercepts = np.zeros((len(lines),), dtype=np.float32)
	max_intercepts = np.zeros((len(lines),), dtype=np.float32)
	
	for line in range(len(lines)):
		min_intercepts[line] = np.inf
		max_intercepts[line] = -np.inf
	
	for p in range(len(points)):
		line = int(points[p, 0])
		intercept = points[p, 1]
		if intercept < min_intercepts[line]:
			min_intercepts[line] = intercept
		if intercept > max_intercepts[line]:
			max_intercepts[line] = intercept
	
	for line in range(len(lines)):
		if max_intercepts[line] == np.inf:
			continue
		length_multiplier = max_intercepts[line] - min_intercepts[line]
		lines[line, 0:3] += lines[line, 3:6] * min_intercepts[line]  # Updates start position of line
		lines[line, 3:6] *= length_multiplier  # Ensures all intercepts are from 0 to 1
		lines[line, 6:9] = lines[line, 3:6] / (lines[line, 3] * lines[line, 3] + lines[line, 4] * lines[line, 4] + lines[line, 5] * lines[line, 5] + 1e-7)
	
	for p in prange(len(points)):
		line = int(points[p, 0])
		points[p, 1] = (points[p, 1] - min_intercepts[line]) / (max_intercepts[line] - min_intercepts[line] + 1e-7)


def initialize_linear(points, lines, topologies, nearby_line):
	lines_cuda = cuda.to_device(lines)
	get_nearest_neighbors(points, lines_cuda, nearby_line)
	features.extend_line_segments[1, 1](points, lines_cuda)
	update_orthogonalization(lines_cuda, topologies)
	lines[:, :] = lines.to_host()


def get_nearest_neighbors(points, lines, nearest):
	threadsperblock = 32
	blockspergrid = (points.size + (threadsperblock - 1)) // threadsperblock
	features.get_nearest_neighbors[blockspergrid, threadsperblock](points, lines, nearest)


def update_orthogonalization(lines, topologies):
	threadsperblock = 32
	blockspergrid = (lines.size + (threadsperblock - 1)) // threadsperblock
	features.update_orthogonalization[blockspergrid, threadsperblock](lines, topologies)


def calculate_topology(topologies, nearest_line):
	threadsperblock = 32
	blockspergrid = (nearest_line.size + (threadsperblock - 1)) // threadsperblock
	features.calculate_topology[blockspergrid, threadsperblock](topologies, nearest_line)


def calculate_linear_updates(nearest_line, updates, update_count):
	threadsperblock = 32
	blockspergrid = (nearest_line.size + (threadsperblock - 1)) // threadsperblock
	features.calculate_linear_updates[blockspergrid, threadsperblock](nearest_line, updates, update_count)


def calculate_topology_updates(topologies, nearest_line, updates, update_count, line=-1):
	threadsperblock = 32
	blockspergrid = (nearest_line.size + (threadsperblock - 1)) // threadsperblock
	features.calculate_topology_updates[blockspergrid, threadsperblock](topologies, line, nearest_line, updates, update_count)


def initialize(points, lines, topologies, nearby_line):
	get_nearest_neighbors(points, lines, nearby_line)
	extend_line_segments(nearby_line, lines)
	update_orthogonalization(lines, topologies)
