from numba import cuda


@cuda.jit
def clear_screen(screen):
	x, y = cuda.grid(2)
	if x < 480 and y < 640:
		screen[x, y, 0] = 0
		screen[x, y, 1] = 0
		screen[x, y, 2] = 0


@cuda.jit
def project_points(points, projection_matrix, projected):
	p = cuda.grid(1)
	point = points[p]
	x = point[0] * projection_matrix[0, 0] + point[1] * projection_matrix[0, 1] + point[2] * projection_matrix[0, 2] + projection_matrix[0, 3]
	y = point[0] * projection_matrix[1, 0] + point[1] * projection_matrix[1, 1] + point[2] * projection_matrix[1, 2] + projection_matrix[1, 3]
	z = point[0] * projection_matrix[2, 0] + point[1] * projection_matrix[2, 1] + point[2] * projection_matrix[2, 2] + projection_matrix[2, 3]
	projected[p, 0] = x / z
	projected[p, 1] = y / z


@cuda.jit
def render_points(points, nearby_lines, projection_matrix, screen):
	p = cuda.grid(1)
	if p >= len(points):
		return
	point = points[p]
	px = point[0]
	py = point[1]
	pz = point[2]
	color = nearby_lines[p, 0]
	x = px * projection_matrix[0, 0] + py * projection_matrix[0, 1] + pz * projection_matrix[0, 2] + projection_matrix[0, 3]
	y = px * projection_matrix[1, 0] + py * projection_matrix[1, 1] + pz * projection_matrix[1, 2] + projection_matrix[1, 3]
	z = px * projection_matrix[2, 0] + py * projection_matrix[2, 1] + pz * projection_matrix[2, 2] + projection_matrix[2, 3]
	u = int(round(x / z))
	v = int(round(y / z))
	if 0 <= u < 640 and 0 <= v < 480:
		if color == 0:
			screen[v, u, 1] = 1
		elif color == 1:
			screen[v, u, 2] = 1
		elif color == 2:
			screen[v, u, 0] = 1
			screen[v, u, 1] = 1
		elif color == 3:
			screen[v, u, 1] = 1
			screen[v, u, 2] = 1
		elif color == 4:
			screen[v, u, 0] = 1
			screen[v, u, 1] = 1
			screen[v, u, 2] = 1
		elif color == 5:
			screen[v, u, 0] = 0.5
			screen[v, u, 1] = 0.5
		else:
			screen[v, u, 1] = 0.5
			screen[v, u, 2] = 0.5
