import numpy as np
import regression_line as rline
import regression_topology as rtop


def evaluate_model(model):
	linear_converged, linear_rmse, linear_iterations = rline.do_linear_regression(model)
	if not linear_converged:
		return None
	
	topology_converged, topology_rmse, topology_iterations = rtop.do_topology_regression(model, min_delta_error=1e-2)
	if not topology_converged:
		return None
	
	# return model, linear_rmse, topology_rmse, linear_iterations, topology_iterations
	return model, linear_rmse, linear_rmse, linear_iterations, 0


def augment_pivot(model, augment_point):
	line_augmented = model.lines[int(augment_point[0])]
	intercept = line_augmented[0:3] + line_augmented[3:6] * augment_point[1]
	
	line_added = np.zeros((9,))
	line_added[0:3] = intercept + augment_point[5:8]
	line_added[3:6] = (line_augmented[0:3] + line_augmented[3:6]) - line_added[0:3]
	
	model_ret = model.copy()
	model_ret.lines[-1, 3:6] = line_added[0:3] - model_ret.lines[-1, 0:3]
	model_ret.add_line(line_added)
	return evaluate_model(model_ret)


def augment_intercept(model, augment_point):
	line_augmented = model.lines[int(augment_point[0])]
	intercept = line_augmented[0:3] + line_augmented[3:6] * augment_point[1]
	
	length = augment_point[5:8]
	line_added = np.zeros((9,))
	line_added[0:3] = intercept
	line_added[3:6] = length
	line_added[6:9] = line_added[3:6] / pow(np.linalg.norm(line_added[3:6]), 2)
	
	model_ret = model.copy()
	model_ret.add_line(line_added)
	return evaluate_model(model_ret)


def augment(model):
	augment_index = int(np.argmax(np.abs(model.nearby_line[:, 2])))
	augment_point = model.nearby_line[augment_index]
	
	pivot_result = augment_pivot(model.copy(), augment_point)
	intercept_result = augment_intercept(model.copy(), augment_point)
	
	return pivot_result if pivot_result[2] < intercept_result[2] else intercept_result
