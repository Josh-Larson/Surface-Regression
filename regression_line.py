import numpy as np
import features


def do_linear_regression(model, learning_rate=0.0, iterations=0, min_delta_error=1e-7):
	"""
	Performs linear regression on the model for some stopping criteria is met
	:param model: the model to run the linear regression on
	:param learning_rate: the learning rate to use (<= 0 automatically determines best LR)
	:param iterations: the maximum number of iterations to run for (<= 0 disables this check)
	:param min_delta_error: the minimum error delta between iterations (<= 0 disables this check)
	:return: a tuple (bool, float64, int) representing the convergence, the final RMSE, and the number of iterations
	"""
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
			converged, rmse, _ = do_linear_regression(test_model, alpha_iteration, 20, 0.0)
			if converged and rmse < best[2]:
				best = test_model, alpha_iteration, rmse
				best_converged = True
			elif best_converged:  # We already have a better solution, no use in exploring further
				break
	return best
