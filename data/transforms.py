import torch
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom

from config.config import cfg

class Pad(object):
	"""
		Adds constant padding to the image boundaries
		Args:
			pad_size (tuple or int): Desired padding. If int, equal padding is used in both dimensions
	"""
	def __init__(self, pad_size=None):
		if pad_size is not None:
			assert isinstance(pad_size, (int, tuple))
			if isinstance(pad_size, int):
				self.pad_size = ((0,0), (pad_size,pad_size), (pad_size,pad_size))
			else:
				assert len(pad_size) == 2
				self.pad_size = ((0,0), (pad_size[0],pad_size[0]), (pad_size[1],pad_size[1]))
		else:
			assert 'PAD' in cfg, "Padding not available"
			self.pad_size = ((0,0),(cfg['PAD'],cfg['PAD']),(cfg['PAD'],cfg['PAD']))
	
	def __call__(self, image):
		# NOTE: Assumed - Image Shape - (C,H,W)
		image = np.pad(image, self.pad_size, 'constant', constant_values=0)
		return image

class MakeSquareAndResize(object):
	"""
		Scales the longer dimension of the image to the size in the config 
		and adds padding to the shorter side to make the image square.
	"""
	def __init__(self):
		pass
	
	def __call__(self, image):
		# NOTE: Assumed - Image Shape - (C,H,W)
		# NOTE: Assumed - cfg includes the target size of the square image
		_, H, W = image.shape
		
		max_dim = max(H, W)
		scale = cfg['SIZE']/max_dim
		newH, newW = int(scale*H), int(scale*W)
		image = zoom(image, (1, scale, scale), order=0, mode='constant', cval=0.0)

		_, H, W = image.shape

		padH_top = (cfg['SIZE'] - H)//2
		padH_bottom = (cfg['SIZE'] - H) - (cfg['SIZE'] - H)//2
		padH = (padH_top, padH_bottom)

		padW_left = (cfg['SIZE'] - W)//2
		padW_right = (cfg['SIZE'] - W) - (cfg['SIZE'] - W)//2
		padW = (padW_left, padW_right)

		pad_size = ((0,0), padH, padW)
		image = np.pad(image, pad_size, 'constant', constant_values=0)
		return image


if __name__ == '__main__':
	# Test MakeSquareAndResize
	img = np.random.random((3, 240, 441))
	transform = MakeSquareAndResize()
	transformed_image = transform(img)
	assert (np.array(transformed_image.shape) == np.array([3,448,448])).all()
	# Test Pad
	transform = Pad()
	transformed_image = transform(img)
	assert (np.array(transformed_image.shape) == np.array([3,240 + 2*cfg['PAD'],441 + 2*cfg['PAD']])).all()