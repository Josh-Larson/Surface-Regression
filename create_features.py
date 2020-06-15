import numpy as np
import math as m
from numba import cuda
from screen import Screen
from model import Model
import cv2
import regression_line as rline
import regression_topology as rtop
import regression_augmentation as raug
import projection
import time

c12_host = np.loadtxt('C12.obj', delimiter=' ').astype(np.float32)
c12 = cuda.to_device(c12_host)

model = Model(cuda.to_device(c12_host))
print("Calculating initial regression...")
rline.do_linear_regression(model)
for i in range(10):
	print("Calculating augmentation %d..." % (i+1))
	model, linear_rmse, topology_rmse, linear_iterations, topology_iterations = raug.augment(model)
	print("Iterations: %d, %d  Topology Error: %f  Linear Error: %f" % (linear_iterations, topology_iterations, topology_rmse, linear_rmse))
	print(model.lines)

print("Finalizing topology...")
rtop.do_topology_regression(model, min_delta_error=1e-4)
print("Topology Error: %f" % model.get_rmse_topology(line=-1))

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
