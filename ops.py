import tensorflow as tf

def conv2d(x, in_channels, output_channels, name, reuse = False):
	'''Convolutional Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [5, 5, in_channels, output_channels], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_channels], initializer = tf.constant_initializer(0.1))

		conv = tf.nn.conv2d(x, w, strides = [1,2,2,1], padding = 'SAME') + b
		return conv

def deconv2d(x, output_shape, name, reuse = False):
	'''Deconvolutional Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [5, 5, output_shape[-1], int(x.get_shape()[-1])], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_shape[-1]], initializer = tf.constant_initializer(0.1))

		deconv = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = [1,2,2,1]) + b
		return deconv

def dense(x, input_dim, output_dim, name, reuse = False):
	'''Fully-connected Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))

		return tf.matmul(x, w) + b