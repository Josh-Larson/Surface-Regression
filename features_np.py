from numba import njit, prange
import numpy as np
import math as m


@njit(parallel=True)
def extend_line_segments(points, lines):
	for line in prange(len(lines)):
		min_intercept = np.inf
		max_intercept = -np.inf
		
		for p in range(len(points)):
			if line != int(points[p, 0]):
				continue
			intercept = points[p, 1]
			if intercept < min_intercept:
				min_intercept = intercept
			if intercept > max_intercept:
				max_intercept = intercept
		
		if max_intercept == np.inf:
			continue
		
		length_multiplier = max_intercept - min_intercept
		lines[line, 0:3] += lines[line, 3:6] * min_intercept  # Updates start position of line
		lines[line, 3:6] *= length_multiplier  # Ensures all intercepts are from 0 to 1
		lines[line, 6:9] = lines[line, 3:6] / (lines[line, 3] * lines[line, 3] + lines[line, 4] * lines[line, 4] + lines[line, 5] * lines[line, 5] + 1e-7)
		
		for p in prange(len(points)):
			if line != int(points[p, 0]):
				continue
			points[p, 1] = (points[p, 1] - min_intercept) / (max_intercept - min_intercept + 1e-7)


@njit(parallel=True)
def get_nearest_neighbors(points, lines, nearest):
	# Point:        x, y, z
	# Line:         x, y, z, lengthX, lengthY, lengthZ, lengthXDot, lengthYDot, lengthZDot, orthoX[0:3], orthoY[0:3], topology[0:6]
	# Nearest Line: line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	for p in prange(len(points)):
		point = points[p]
		previous_distance = np.inf
		for i, line in enumerate(lines):
			pt_vec_x = point[0] - line[0]
			pt_vec_y = point[1] - line[1]
			pt_vec_z = point[2] - line[2]
			t = max(0., min(1., pt_vec_x * line[6] + pt_vec_y * line[7] + pt_vec_z * line[8]))
			dx = (t * line[3] - pt_vec_x)
			dy = (t * line[4] - pt_vec_y)
			dz = (t * line[5] - pt_vec_z)
			dist = dx*dx + dy*dy + dz*dz
			
			projected_x = (dx * line[9] + dy * line[10] + dz * line[11])
			projected_y = (dx * line[12] + dy * line[13] + dz * line[14])
			angle = m.atan2(projected_y, projected_x) / m.pi  # Normalized between -1 and 1
			
			predicted_distance = line[21] * t * t * angle * angle + \
			                     line[20] * t * t + \
			                     line[19] * angle * angle + \
			                     line[18] * angle * t + \
			                     line[17] * t + \
			                     line[16] * angle + \
			                     line[15]
			topology_error = (m.sqrt(dist) - predicted_distance)**2

			if i == 0 or dist < nearest[p, 2]:
				previous_distance = dist
				nearest[p, 0] = i
				nearest[p, 1] = t
				nearest[p, 2] = dist
				nearest[p, 3] = projected_x  # Angle
				nearest[p, 4] = predicted_distance  # Distance Error
				nearest[p, 5] = dx
				nearest[p, 6] = dy
				nearest[p, 7] = dz


@njit(parallel=True)
def update_orthogonalization(lines):
	for line_idx in prange(len(lines)):
		line = lines[line_idx]
		line_definition = line[3:6]
		line_magnitude = m.sqrt(line[3]*line[3] + line[4]*line[4] + line[5]*line[5])
		angle1 = m.acos(line[3] / line_magnitude)
		angle2 = m.acos(line[4] / line_magnitude)
		initial_ortho_x = 0.
		initial_ortho_y = 0.
		initial_ortho_z = 0.
		if 45 <= abs(angle1) <= 135:
			initial_ortho_x = 1.
		elif 45 <= abs(angle2) <= 135:
			initial_ortho_y = 1.
		else:
			initial_ortho_z = 1.
			# Might need to add an assertion to ensure that this angle isn't invalid either, but I hope that's impossible
		# Cross Product Time
		# First Orthogonal Axis (x)
		line[9] = line_definition[1] * initial_ortho_z + line_definition[2] * initial_ortho_y
		line[10] = line_definition[2] * initial_ortho_x + line_definition[0] * initial_ortho_z
		line[11] = line_definition[0] * initial_ortho_y + line_definition[1] * initial_ortho_x
		x_magnitude = m.sqrt(line[9]*line[9] + line[10]*line[10] + line[11]*line[11]) + 1e-7
		line[9] /= x_magnitude
		line[10] /= x_magnitude
		line[11] /= x_magnitude
		# Second Orthogonal Axis (y)
		line[12] = line_definition[1] * line[11] + line_definition[2] * line[10]
		line[13] = line_definition[2] * line[9] + line_definition[0] * line[11]
		line[14] = line_definition[0] * line[10] + line_definition[1] * line[9]
		y_magnitude = m.sqrt(line[12]*line[12] + line[13]*line[13] + line[14]*line[14]) + 1e-7
		line[12] /= y_magnitude
		line[13] /= y_magnitude
		line[14] /= y_magnitude


@njit
def calculate_linear_updates(nearest_line, updates, update_count):
	# Nearest Line:     line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	for point in nearest_line:
		line_index = int(point[0])
		update_line = updates[line_index]
		update_count[line_index] += 1
		
		intercept = point[1]
		for i in range(3):
			val = point[i + 5]
			update_line[i] += -2 * val
			update_line[i + 3] += -2 * val * intercept  # x/y/z * intercept
		update_line[6] += point[2]  # actual_distance
		# 	count += 1
		# update_count[line_index] = count
		# updates[line_index, 0:7] = update_line


@njit
def calculate_topology_updates(lines, line_enabled, nearest_line, updates, update_count):
	# Line:         x, y, z, lengthX, lengthY, lengthZ, lengthXDot, lengthYDot, lengthZDot, orthoX[0:3], orthoY[0:3], topology[0:6]
	# Nearest Line: line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	for point in nearest_line:
		line_index = int(point[0])
		# if line_enabled >= 0 and line_enabled != line_index:
		# 	continue
		
		update_line = updates[line_index]
		update_count[line_index] += 1
		
		intercept = point[1]
		angle = point[3]
		predicted_distance = point[4]
		
		actual_distance = point[2]
		distance_error = actual_distance - predicted_distance
		
		# update_line[7 + 0] += -2 * distance_error
		# update_line[7 + 1] += -2 * distance_error * angle
		# update_line[7 + 2] += -2 * distance_error * intercept
		# update_line[7 + 3] += -2 * distance_error * angle * intercept
		# update_line[7 + 4] += -2 * distance_error * angle * angle
		# update_line[7 + 5] += -2 * distance_error * intercept * intercept
		# update_line[7 + 6] += -2 * distance_error * angle * angle * intercept * intercept
		# update_line[7 + 7] += distance_error * distance_error
