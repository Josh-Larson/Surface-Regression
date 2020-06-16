from numba import cuda
import numpy as np
import projection_cuda as projection
import math as m


def clear_screen(screen):
	threadsperblock = 32
	blockspergrid_x = (640 + (threadsperblock - 1)) // threadsperblock
	blockspergrid_y = (480 + (threadsperblock - 1)) // threadsperblock
	projection.clear_screen[(blockspergrid_y, blockspergrid_x), (threadsperblock, threadsperblock)](screen)


def project_points(points, projection_matrix):
	projected = np.zeros((len(points), 2), dtype=np.float32)
	threadsperblock = 32
	blockspergrid = (len(points) + (threadsperblock - 1)) // threadsperblock
	projection.project_points[blockspergrid, threadsperblock](points, projection_matrix, projected)
	return projected


def render_points(points, nearby_lines, projection_matrix, screen):
	threadsperblock = 32
	blockspergrid = (len(points) + (threadsperblock - 1)) // threadsperblock
	projection.render_points[blockspergrid, threadsperblock](points, nearby_lines, projection_matrix, screen)


def createRot(RPY, degrees=False):
	if degrees:
		i_yaw = m.radians(RPY[2])
		i_pitch = m.radians(RPY[1])
		i_roll = m.radians(RPY[0])
	else:
		i_yaw = RPY[2]
		i_pitch = RPY[1]
		i_roll = RPY[0]
	
	R_yaw = np.array([[m.cos(i_yaw), 0., -m.sin(i_yaw)], [0., 1., 0.], [m.sin(i_yaw), 0., m.cos(i_yaw)]])
	R_pitch = np.array([[1., 0., 0.], [0., m.cos(i_pitch), m.sin(i_pitch)], [0., -m.sin(i_pitch), m.cos(i_pitch)]])
	R_roll = np.array([[m.cos(i_roll), m.sin(i_roll), 0.], [-m.sin(i_roll), m.cos(i_roll), 0.], [0., 0., 1.]])
	return np.dot(R_roll, np.dot(R_pitch, R_yaw))


def create_camera(location, look_at, width, height):
	up = np.array([0., 0., 1.], dtype=np.float32)
	vec_x = look_at - location
	vec_x /= np.linalg.norm(vec_x)
	vec_y = np.cross(up, vec_x)
	vec_y /= np.linalg.norm(vec_y)
	vec_z = np.cross(vec_x, vec_y)
	vec_z /= np.linalg.norm(vec_z)
	projection_matrix = np.zeros((3, 4), dtype=np.float32)
	projection_matrix[0, :3] = vec_x
	projection_matrix[1, :3] = vec_y
	projection_matrix[2, :3] = vec_z
	projection_matrix[0:3, 3] = projection_matrix[:3, :3] @ -location
	remap = np.array([[0, -1, 0],
	                  [0, 0, -1],
	                  [1, 0, 0]])
	camera_matrix = np.array([[width, 0, width/2],
	                          [0, height, height/2],
	                          [0, 0, 1]], dtype=np.float32)
	return camera_matrix @ (remap @ projection_matrix)
