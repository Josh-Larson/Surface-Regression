import numpy as np
import math as m
from numba import cuda
from screen import Screen
from model import Model
import cv2
import features
import projection
import time

c12_host = np.loadtxt('C12.obj', delimiter=' ').astype(np.float32)
c12 = cuda.to_device(c12_host)

model = Model(c12_host)
model.linear_regression()
for i in range(10):
	polar_error, linear_error, iterations, update_difference = model.augment_lines()
	print("Iterations: %d  Polar Error: %f  Linear Error: %f  Delta Error: %f" % (iterations, model.get_rmse_polar(), model.get_rmse_linear(), update_difference))
	print(model.lines)

# Render
screen = Screen()
end = -1
iter = 0
while end == -1:
	screen.render(c12, (iter % 360) / 360 * (2 * m.pi))
	display = screen.get_screen()
	for line in model.lines:
		start = screen.project_point(line[0:3])
		end = screen.project_point(line[0:3] + line[3:6])
		cv2.line(display, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), thickness=3)
	cv2.imshow("C12", display)
	end = cv2.waitKey(5)
	iter += 1
