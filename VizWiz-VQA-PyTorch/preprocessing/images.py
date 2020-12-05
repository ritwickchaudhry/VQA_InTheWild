import os

import sys 
sys.path.append('./datasets')

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from custom_transforms import RandomResizedCrop
import numpy as np

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ImageDataset(data.Dataset):

	def __init__(self, path, config, transform=None, extreme_transform=None):
		self.path = path
		self.transform = transform

		# Extreme parameters
		self.extreme_transform = extreme_transform
		self.extreme_perturb = config['images']['augmentation']['extreme_perturb']
		self.extreme_p = config['images']['augmentation']['extreme_p']

		# Load the paths to the images available in the folder
		self.image_names = self._load_img_paths()

		if len(self.image_names) == 0:
			raise (RuntimeError("Found 0 images in " + path + "\n"
															  "Supported image extensions are: " + ",".join(
				IMG_EXTENSIONS)))
		else:
			print('Found {} images in {}'.format(len(self), self.path))

	def __getitem__(self, index):
		item = {}
		item['name'] = self.image_names[index]
		item['path'] = os.path.join(self.path, item['name'])

		# Use PIL to load the image
		item['visual'] = Image.open(item['path']).convert('RGB')

		extreme_transformed = False
		if self.extreme_perturb:
			p = np.random.randn()
			if p <= self.extreme_p:
				extreme_transformed = True
				if self.extreme_transform is not None: item['visual'] = self.extreme_transform(item['visual'])

		if self.transform is not None and not extreme_transformed:
			item['visual'] = self.transform(item['visual'])

		item['extreme_transformed'] = extreme_transformed
		return item

	@property
	def get_image_names(self):
		return self.image_names

	def __len__(self):
		return len(self.image_names)

	def _load_img_paths(self):
		images = []
		for name in os.listdir(self.path): # TODO: Add train / val split
			if is_image_file(name):
				images.append(name)
		print(len(images))
		return images

def get_extreme_transform(config):
	img_config = config['images']
	img_size = img_config['img_size']
	all_transforms = []
	augmentation_config = img_config['augmentation']

	all_transforms.append(
		RandomResizedCrop(
			img_size,
			scale = augmentation_config['e_crop_scale'],
			ratio = augmentation_config['e_crop_ratio']
		))

	all_transforms.append(
		transforms.GaussianBlur(augmentation_config['e_blur_size'], augmentation_config['e_blur_sigma'])
		)

	all_transforms += [
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4711275, 0.44754702, 0.40802705],
							std=[0.24383631, 0.23895378, 0.24194406]),
	]

	all_transforms.append(
		transforms.RandomErasing(
			p=augmentation_config['e_erasing_probability'], 
			scale=augmentation_config['e_erasing_scale']
		)
	)

	return transforms.Compose(all_transforms)

def get_transform(config):
	img_config = config['images']
	img_size = img_config['img_size']
	all_transforms = []
	augmentation_config = img_config['augmentation']

	if augmentation_config['do_crop']:
		all_transforms.append(
			RandomResizedCrop(
				img_size,
				scale = augmentation_config['crop_scale'],
				ratio = augmentation_config['crop_ratio']
			))
	else:
		all_transforms.append(transforms.Resize(img_size))

	if augmentation_config['do_blur']:
		all_transforms.append(
			transforms.GaussianBlur(augmentation_config['blur_size'], augmentation_config['blur_sigma'])
			)

	all_transforms += [
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4711275, 0.44754702, 0.40802705],
							std=[0.24383631, 0.23895378, 0.24194406]),
	]

	if augmentation_config['do_erasing']:
		all_transforms.append(
			transforms.RandomErasing(
				p=augmentation_config['erasing_probability'], 
				scale=augmentation_config['erasing_scale']
			)
		)

	return transforms.Compose(all_transforms)