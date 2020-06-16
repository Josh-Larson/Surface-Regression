import numpy as np
import math as m
from numba import cuda
import cv2
import features
import projection
import time


class Screen:
	def __init__(self):
		# TODO: Allow different sizes
		self.screen = np.empty((480, 640, 3), dtype=np.float32)
		self.projection_matrix = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
		self.render_time = 0.
	
	def get_screen(self):
		return self.screen
	
	def project_point(self, point):
		projected = self.projection_matrix @ np.concatenate((point, [1]))
		return projected[:2] / projected[2]
	
	def render(self, points, nearby_lines, angle):
		start_time = time.time_ns()
		self.screen[:, :] = 0
		x = m.cos(angle) * 25
		y = m.sin(angle) * 25
		self.projection_matrix = projection.create_camera(np.array([x, y, 25.]), np.zeros((3,)), 640, 480)
		camera = cuda.to_device(self.projection_matrix)
		screen_device = cuda.to_device(self.screen)
		projection.clear_screen(screen_device)
		projection.render_points(points, nearby_lines, camera, screen_device)
		self.screen = screen_device.copy_to_host()
		end_time = time.time_ns()
		self.render_time = (end_time - start_time) / 1000000
