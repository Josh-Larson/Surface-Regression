import numpy as np
import features
import regression_line as rline


def do_topology_regression(model, learning_rate=0., line=-1, iterations=0, min_delta_error=1e-7):
	"""
	Performs topology regression on the model for some stopping criteria is met
	:param model: the model to run the topology regression on
	:param learning_rate: the learning rate to use (<= 0 automatically determines best LR)
	:param line: the specific line to update the topology for (< 0 indicates all lines)
	:param iterations: the maximum number of iterations to run for (<= 0 disables this check)
	:param min_delta_error: the minimum error delta between iterations (<= 0 disables this check)
	:return: a tuple (bool, float64, int) representing the convergence, the final RMSE, and the number of iterations
	"""
	if learning_rate <= 0:
		learning_rate = get_best_learning_rates(model)
		print(learning_rate)
	iteration = 0
	previous_error = np.inf
	while iterations <= 0 or iteration < iterations:
		# Calculates linear regression updates
		model.updates.clear()
		features.calculate_topology_updates(model.lines, model.nearby_line, model.updates.updates, model.updates.update_count, line)
		model.remove_unused_lines()
		model.updates.adjust_by_count()
		if line < 0:
			assert not (model.updates.update_count <= 0).any()
		else:
			assert model.updates.update_count[line] != 0
		# Failure case: numerical errors/divergence
		if np.isnan(model.updates.updates).any() or np.isinf(model.updates.updates).any():
			print("Topology: NaN/Inf")
			return False, 0.0, iteration
		# Update parameters
		if line >= 0:
			model.lines[line, 15:22] += learning_rate * model.updates.updates[line, 7:14]
		else:
			model.lines[:, 15:22] += learning_rate * model.updates.updates[:, 7:14]
		# Check delta error stopping criteria
		current_error = model.get_rmse_topology(line)
		delta_error = previous_error - current_error
		previous_error = current_error
		# print("%03d: %.20f" % (iteration, previous_error))
		if delta_error < 0:
			# print("Topology: Diverge  %s" % str(delta_error))
			return False, current_error, iteration  # Diverging
		if delta_error < min_delta_error:
			return True, current_error, iteration  # Met min_delta_error stopping criteria
		iteration += 1
	return True, previous_error, iteration


def get_best_learning_rate(model, line):
	"""
	Calculates the best learning rate via trial and error
	:param model: the model to start from
	:param line: the specific line to find the learning rate for
	:return: a tuple with the best model so far, the learning rate, and the final RMSE
	"""
	best_converged = False
	best = model, 1e-5, np.inf
	for learning_rate_exp in range(-1, -6, -1):
		alpha = pow(10, learning_rate_exp)
		for i in range(2):
			alpha_iteration = alpha * 3 if i == 0 else alpha
			test_model = model.copy()
			converged, rmse, _ = do_topology_regression(test_model, alpha_iteration, line, 20, 0)
			if converged and rmse < best[2]:
				best = test_model, alpha_iteration, rmse
				best_converged = True
			elif best_converged:  # We already have a better solution, no use in exploring further
				break
	return best


def get_best_learning_rates(model):
	# learning_rate = np.zeros((len(model.lines), 1), dtype=np.float64)
	# for line in range(len(model.lines)):
	# 	learning_rate[line] = get_best_learning_rate(model, line)[1]
	# return learning_rate
	return get_best_learning_rate(model.copy(), -1)[1]
