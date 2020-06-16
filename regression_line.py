import numpy as np
import features


def do_linear_regression_manual(model, learning_rate, iterations, min_delta_error):
	if learning_rate <= 0:
		learning_rate = get_best_learning_rate(model)[1]
	iteration = 0
	previous_error = np.inf
	while iterations <= 0 or iteration < iterations:
		# Calculates nearest neighbor, line segment length, and orthogonalization
		features.initialize(model.points, model.lines, model.nearby_line)
		# Calculates linear regression updates
		model.updates.clear()
		features.calculate_linear_updates(model.nearby_line, model.updates.updates, model.updates.update_count)
		model.remove_unused_lines()
		model.updates.adjust_by_count()
		# Failure case: numerical errors/divergence
		if np.isnan(model.updates.updates).any() or np.isinf(model.updates.updates).any():
			return False, 0.0, iteration
		# Update parameters
		model.lines[:, 0:6] += learning_rate * model.updates.updates[:, 0:6]
		# Check delta error stopping criteria
		current_error = model.get_rmse_linear()
		delta_error = previous_error - current_error
		previous_error = current_error
		if delta_error < 0:
			return False, current_error, iteration  # Diverging
		if delta_error < min_delta_error:
			return True, current_error, iteration  # Met min_delta_error stopping criteria
		iteration += 1
	return True, previous_error, iteration


def do_topologies_intersect(line1, line2):
	w0 = line1[0:3] - line2[0:3]
	a = np.dot(line1[6:9], line1[6:9])
	b = np.dot(line1[6:9], line2[6:9])
	c = np.dot(line2[6:9], line2[6:9])
	d = np.dot(line1[6:9], w0)
	e = np.dot(line2[6:9], w0)
	denom = a * c - b * b
	if denom == 0:
		return True, 0., 0.
	l1t = max(0., min(np.dot(line1[3:6], line1[6:9]), (b * e - c * d) / denom))
	l2t = max(0., min(np.dot(line2[3:6], line2[6:9]), (a * e - b * d) / denom))
	p1 = line1[0:3] + l1t * line1[6:9]
	p2 = line2[0:3] + l2t * line2[6:9]
	distance = np.linalg.norm(p1 - p2)
	d1 = features.surface_regression(line1, p2)
	d2 = features.surface_regression(line2, p1)
	
	return d1 + d2 >= distance, l1t, l2t


def do_linear_regression_pca(model, remove_unused=True):
	features.get_nearest_neighbors(model.points, model.lines, model.nearby_line)
	for line_idx, line in enumerate(model.lines):
		points = model.points[model.nearby_line[:, 0] == line_idx]
		# min_intercept_cap = -np.inf
		# max_intercept_cap = np.inf
		#
		# for prev_line in range(line_idx):
		# 	intersects, l1t, l2t = do_topologies_intersect(line, model.lines[prev_line])
		# 	if intersects:
		# 		if l1t >= 0:
		# 			max_intercept_cap = min(max_intercept_cap, l1t)
		# 		else:
		# 			min_intercept_cap = max(min_intercept_cap, l1t)
		
		# print(min_intercept_cap <= np.dot(points, line[6:9]).flatten() <= max_intercept_cap)
		# points = points[np.where(min_intercept_cap <= np.dot(points, line[6:9]) <= max_intercept_cap]
		model.updates.update_count[line_idx] = len(points)
		if len(points) == 0:
			continue
		
		points_average = points.mean(axis=0)
		points = points - points_average
		C = np.dot(points.transpose(), points) / (len(points) - 1)
		eigen_vals, eigen_vecs = np.linalg.eig(C)
		line[6:15] = eigen_vecs.flatten()
		min_intercept = np.inf
		max_intercept = -np.inf
		for p_index, p in enumerate(points):
			intercept = np.dot(p, line[6:9])
			min_intercept = min(min_intercept, intercept)
			max_intercept = max(max_intercept, intercept)
		length = max_intercept - min_intercept
		start = points_average + line[6:9] * min_intercept
		line[0:3] = start
		line[3:6] = length * line[6:9]
		line[6:9] /= length
		for p in range(len(model.nearby_line)):
			if line_idx != int(model.nearby_line[p, 0]):
				continue
			model.nearby_line[p, 1] = np.dot(model.points[p, 0:3] - line[0:3], line[6:9])
	if remove_unused:
		model.remove_unused_lines()
	return True, model.get_rmse_linear(), 0


def do_linear_regression(model, learning_rate=0.0, iterations=0, min_delta_error=1e-7):
	"""
	Performs linear regression on the model for some stopping criteria is met
	:param model: the model to run the linear regression on
	:param learning_rate: the learning rate to use (<= 0 automatically determines best LR)
	:param iterations: the maximum number of iterations to run for (<= 0 disables this check)
	:param min_delta_error: the minimum error delta between iterations (<= 0 disables this check)
	:return: a tuple (bool, float64, int) representing the convergence, the final RMSE, and the number of iterations
	"""
	return do_linear_regression_pca(model)


def get_best_learning_rate(model):
	"""
	Calculates the best learning rate via trial and error
	:param model: the model to start from
	:return: a tuple with the best model so far, the learning rate, and the final RMSE
	"""
	best_converged = False
	best = model, 1e-5, np.inf
	for learning_rate_exp in range(-1, -6, -1):
		alpha = pow(10, learning_rate_exp)
		for i in range(2):
			alpha_iteration = alpha * 3 if i == 0 else alpha
			test_model = model.copy()
			converged, rmse, _ = do_linear_regression_manual(test_model, alpha_iteration, 20, 0.0)
			if converged and rmse < best[2]:
				best = test_model, alpha_iteration, rmse
				best_converged = True
			elif best_converged:  # We already have a better solution, no use in exploring further
				break
	return best
