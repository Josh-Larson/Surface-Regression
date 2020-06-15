from numba import cuda
import numpy as np
import math as m


@cuda.jit
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
	
	for p in range(len(points)):
		line = int(points[p, 0])
		points[p, 1] = (points[p, 1] - min_intercepts[line]) / (max_intercepts[line] - min_intercepts[line] + 1e-7)


@cuda.jit
def get_nearest_neighbors(points, lines, nearest):
	# Point:        x, y, z
	# Line:         x, y, z, lengthX, lengthY, lengthZ, lengthXDot, lengthYDot, lengthZDot, orthoX[0:3], orthoY[0:3], topology[0:6]
	# Nearest Line: line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	p = cuda.grid(1)
	if p >= len(points):
		return
	point = points[p]
	for i, line in enumerate(lines):
		pt_vec_x = point[0] - line[0]
		pt_vec_y = point[1] - line[1]
		pt_vec_z = point[2] - line[2]
		t = pt_vec_x * line[6] + pt_vec_y * line[7] + pt_vec_z * line[8]
		dx = (t * line[3] - pt_vec_x)
		dy = (t * line[4] - pt_vec_y)
		dz = (t * line[5] - pt_vec_z)
		dist = dx*dx + dy*dy + dz*dz
		if i == 0 or dist < nearest[p, 2]:
			nearest[p, 0] = i
			nearest[p, 1] = t
			nearest[p, 2] = dist
			nearest[p, 3] = 0.  # Angle
			nearest[p, 4] = 0.  # Distance Error
			nearest[p, 5] = dx
			nearest[p, 6] = dy
			nearest[p, 7] = dz


@cuda.jit
def update_orthogonalization(lines):
	# Orthogonalization: 1=x  2=y  3=z
	l = cuda.grid(1)
	if l >= len(lines):
		return
	line = lines[l]
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
	topology = line[9:15]
	topology[0] = line_definition[1] * initial_ortho_z + line_definition[2] * initial_ortho_y
	topology[1] = line_definition[2] * initial_ortho_x + line_definition[0] * initial_ortho_z
	topology[2] = line_definition[0] * initial_ortho_y + line_definition[1] * initial_ortho_x
	x_magnitude = m.sqrt(topology[0]*topology[0] + topology[1]*topology[1] + topology[2]*topology[2]) + 1e-7
	topology[0] /= x_magnitude
	topology[1] /= x_magnitude
	topology[2] /= x_magnitude
	# Second Orthogonal Axis (y)
	topology[3] = line_definition[1] * topology[2] + line_definition[2] * topology[1]
	topology[4] = line_definition[2] * topology[0] + line_definition[0] * topology[2]
	topology[5] = line_definition[0] * topology[1] + line_definition[1] * topology[0]
	y_magnitude = m.sqrt(topology[3]*topology[3] + topology[4]*topology[4] + topology[5]*topology[5]) + 1e-7
	topology[3] /= y_magnitude
	topology[4] /= y_magnitude
	topology[5] /= y_magnitude


@cuda.jit
def calculate_linear_updates(nearest_line, updates, update_count):
	# Nearest Line:     line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	p = cuda.grid(1)
	if p >= len(nearest_line):
		return
	point = nearest_line[p]
	line_index = int(point[0])
	actual_distance = point[2]
	
	intercept = point[1]
	cuda.atomic.add(updates[line_index], 0, -2 * point[5])
	cuda.atomic.add(updates[line_index], 1, -2 * point[6])
	cuda.atomic.add(updates[line_index], 2, -2 * point[7])
	cuda.atomic.add(updates[line_index], 3, -2 * point[5] * intercept)
	cuda.atomic.add(updates[line_index], 4, -2 * point[6] * intercept)
	cuda.atomic.add(updates[line_index], 5, -2 * point[7] * intercept)
	cuda.atomic.add(updates[line_index], 6, actual_distance)
	cuda.atomic.add(update_count[line_index], 0, 1)


@cuda.jit
def calculate_topology_updates(lines, line_enabled, nearest_line, updates, update_count):
	# Line:         x, y, z, lengthX, lengthY, lengthZ, lengthXDot, lengthYDot, lengthZDot, orthoX[0:3], orthoY[0:3], topology[0:6]
	# Nearest Line: line_idx, intercept, distance, angle, distance_error, dx, dy, dz
	p = cuda.grid(1)
	if p >= len(nearest_line):
		return
	point = nearest_line[p]
	line_index = int(point[0])
	if line_enabled >= 0 and line_enabled != line_index:
		return
	line = lines[int(point[0])]
	actual_distance = point[2]
	projected_x = (point[5]*line[9] + point[6]*line[10] + point[7]*line[11]) / actual_distance
	projected_y = (point[5]*line[12] + point[6]*line[13] + point[7]*line[14]) / actual_distance
	angle = m.atan2(projected_y, projected_x) / m.pi  # Normalized between -1 and 1
	intercept = point[1]
	
	predicted_distance = line[21] * intercept * intercept * angle * angle + \
	                     line[20] * intercept * intercept + \
	                     line[19] * angle * angle + \
	                     line[18] * angle * intercept + \
	                     line[17] * intercept + \
	                     line[16] * angle + \
	                     line[15]
	distance_error = predicted_distance - actual_distance
	point[3] = angle
	point[4] = actual_distance - predicted_distance
	
	cuda.atomic.add(updates[line_index], 7+0, -2 * distance_error)
	cuda.atomic.add(updates[line_index], 7+1, -2 * distance_error * angle)
	cuda.atomic.add(updates[line_index], 7+2, -2 * distance_error * intercept)
	cuda.atomic.add(updates[line_index], 7+3, -2 * distance_error * angle * intercept)
	cuda.atomic.add(updates[line_index], 7+4, -2 * distance_error * angle * angle)
	cuda.atomic.add(updates[line_index], 7+5, -2 * distance_error * intercept * intercept)
	cuda.atomic.add(updates[line_index], 7+6, -2 * distance_error * angle * angle * intercept * intercept)
	cuda.atomic.add(updates[line_index], 7+7, distance_error ** 2)
	cuda.atomic.add(update_count[line_index], 0, 1)
