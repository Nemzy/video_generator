from PIL import Image
import numpy as np
import os

class DataSet(object):
	def __init__(self, images):

		self._num_examples = len(images)
		self._images = images
		self._index_in_epoch = 0
		self._epochs_completed = 0

	def images(self):
		'''Returns images.'''
		return self._images

	def num_examples(self):
		'''Returns number of images.'''
		return self._num_examples

	def epochs_completed(self):
		'''Returns number of completed epochs.'''
		return self._epochs_completed

	def next_batch(self, batch_size):
		'''Return the next `batch_size` images from the data set.'''
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:

			self._epochs_completed += 1

			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch

		return self._images[start:end]

def read_images(filenames):
	'''Reads images from file names'''
	images = []
	for file in filenames:
		img = Image.open(file)
		image = np.array(img, dtype = np.float32)
		image = np.multiply(image, 1.0 / 255.0)
		images.append(image)

	return images

def read_dataset(path):
	'''Creates data set'''
	dirpath, dirnames, filenames = next(os.walk(path))
	images = read_images([os.path.join(dirpath, filename) for filename in filenames])
	images = np.array(images, dtype = np.float32)

	return DataSet(images)

if __name__ == '__main__':
	ds = read_dataset('data/frames')
	print 'Shape:',ds.images().shape
	print 'Memory size:', ds.images().nbytes / (1024.0 * 1024.0), 'MB'
	print 'Batch shape:', ds.next_batch(100).shape