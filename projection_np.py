from numba import njit, prange


@njit
def project_points(points, projection_matrix, projected):
	for p in prange(len(points)):
		point = points[p]
		x = point[0] * projection_matrix[0, 0] + point[1] * projection_matrix[0, 1] + point[2] * projection_matrix[0, 2] + projection_matrix[0, 3]
		y = point[0] * projection_matrix[1, 0] + point[1] * projection_matrix[1, 1] + point[2] * projection_matrix[1, 2] + projection_matrix[1, 3]
		z = point[0] * projection_matrix[2, 0] + point[1] * projection_matrix[2, 1] + point[2] * projection_matrix[2, 2] + projection_matrix[2, 3]
		projected[p, 0] = x / z
		projected[p, 1] = y / z
