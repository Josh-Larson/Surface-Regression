import features_cuda
import features_np


def initialize_linear(points, lines, nearby_line):
	get_nearest_neighbors(points, lines, nearby_line)
	features_np.extend_line_segments(points, lines)
	update_orthogonalization(lines)


def get_nearest_neighbors(points, lines, nearest):
	threadsperblock = 32
	blockspergrid = (len(points) + (threadsperblock - 1)) // threadsperblock
	features_cuda.get_nearest_neighbors[blockspergrid, threadsperblock](points, lines, nearest)
	# features_np.get_nearest_neighbors(points, lines, nearest)


def update_orthogonalization(lines):
	# threadsperblock = 32
	# blockspergrid = (lines.size + (threadsperblock - 1)) // threadsperblock
	# features_cuda.update_orthogonalization[blockspergrid, threadsperblock](lines)
	features_np.update_orthogonalization(lines)


def calculate_linear_updates(nearest_line, updates, update_count):
	# threadsperblock = 32
	# blockspergrid = (nearest_line.size + (threadsperblock - 1)) // threadsperblock
	# features_cuda.calculate_linear_updates[blockspergrid, threadsperblock](nearest_line, updates, update_count)
	features_np.calculate_linear_updates(nearest_line, updates, update_count)


def calculate_topology_updates(lines, nearest_line, updates, update_count, line=-1):
	# threadsperblock = 32
	# blockspergrid = (nearest_line.size + (threadsperblock - 1)) // threadsperblock
	# features_cuda.calculate_topology_updates[blockspergrid, threadsperblock](lines, line, nearest_line, updates, update_count)
	features_np.calculate_topology_updates(lines, line, nearest_line, updates, update_count)


def initialize(points, lines, nearby_line):
	get_nearest_neighbors(points, lines, nearby_line)
	features_np.extend_line_segments(nearby_line, lines)
	update_orthogonalization(lines)
