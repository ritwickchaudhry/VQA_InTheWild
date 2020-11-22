import os

import sys 
sys.path.append('./datasets')

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ImageDataset(data.Dataset):

	def __init__(self, path, transform=None):
		self.path = path
		self.transform = transform

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
		if self.transform is not None:
			item['visual'] = self.transform(item['visual'])

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


def get_transform(config):
	img_config = config['images']
	img_size = img_config['img_size']
	all_transforms = []
	if img_config['augmentation']['do_crop']:
		all_transforms.append(
			# transforms.RandomResizedCrop( 
			RandomResizedCrop(
				img_size,
				scale = img_config['augmentation']['crop_scale'],
				ratio = img_config['augmentation']['crop_ratio']
			))
	else:
		all_transforms.append(transforms.Resize(img_size))
	all_transforms += [
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4774, 0.4510, 0.4103],
							std=[0.2741, 0.2692, 0.2841]),
	]
	return transforms.Compose(all_transforms)