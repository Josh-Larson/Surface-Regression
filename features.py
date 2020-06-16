import features_cuda
import features_np
import math as m


def surface_regression(line, point):
	pt_vec_x = point[0] - line[0]
	pt_vec_y = point[1] - line[1]
	pt_vec_z = point[2] - line[2]
	t = pt_vec_x * line[6] + pt_vec_y * line[7] + pt_vec_z * line[8]
	dx = (t * line[3] - pt_vec_x)
	dy = (t * line[4] - pt_vec_y)
	dz = (t * line[5] - pt_vec_z)
	
	projected_x = (dx * line[9] + dy * line[10] + dz * line[11])
	projected_y = (dx * line[12] + dy * line[13] + dz * line[14])
	angle = m.atan2(projected_y, projected_x) / m.pi  # Normalized between -1 and 1
	
	return line[21] * t * t * angle * angle + \
	       line[20] * t * t + \
	       line[19] * angle * angle + \
	       line[18] * angle * t + \
	       line[17] * t + \
	       line[16] * angle + \
	       line[15]


def initialize_linear(points, lines, nearby_line):
	get_nearest_neighbors(points, lines, nearby_line)
	features_np.extend_line_segments(points, lines)
	update_orthogonalization(lines)


def get_nearest_neighbors(points, lines, nearest):
	# threadsperblock = 32
	# blockspergrid = (len(points) + (threadsperblock - 1)) // threadsperblock
	# features_cuda.get_nearest_neighbors[blockspergrid, threadsperblock](points, lines, nearest)
	features_np.get_nearest_neighbors(points, lines, nearest)


def extend_line_segments(nearest, lines):
	features_np.extend_line_segments(nearest, lines)


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
	extend_line_segments(nearby_line, lines)
	update_orthogonalization(lines)
