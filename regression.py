import model
import regression_line as rline
import regression_topology as rtop
import regression_augmentation as raug
from numba import cuda


class Regression:
	def __init__(self, points):
		self.model = model.Model(cuda.to_device(points))
		self.rline = rline.RegressionLine(self.model)
		self.rtop = rtop.RegressionTopology(self.model)
		self.raug = raug.RegressionAugmentation(self.model)
