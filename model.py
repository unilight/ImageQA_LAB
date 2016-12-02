import tensorflow as tf
import math

class Model:
	def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def random_bias(self, dim, name=None):
		return tf.Variable(tf.zeros([dim]), name=name)

	def __init__(self, options):
		with tf.device('/cpu:0'):
			self.options = options

			
