import os
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets import vqa_dataset
from torchvision import transforms

def revert_transform(image_tensor):
	inv_normalize = transforms.Normalize(
    mean=[-0.4774/0.2741, -0.4510/0.2692, -0.4103/0.2841],
    std=[1/0.2741, 1/0.2692, 1/0.2841]
	)
	inv_tensor = inv_normalize(image_tensor)
	to_pil = transforms.ToPILImage()
	return to_pil(inv_tensor)

if __name__ == "__main__":
	# Load config yaml file
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_config', default='config/default.yaml', type=str,
						help='path to a yaml config file')
	args = parser.parse_args()

	with open(args.path_config, 'r') as handle:
		config = yaml.load(handle)
	
	split='train'
	dataset = vqa_dataset.VQADataset(
        config,
        split
    )
	i = 0
	item = dataset[i]
	img_dir = os.path.join(config['images']['dir'], split)
	img_path = os.path.join(img_dir, item['img_name'])
	original_image = Image.open(img_path).convert('RGB')
	transformed_image = revert_transform(item['visual'])
	fig, axes = plt.subplots(1,2)
	axes[0].imshow(original_image)
	axes[0].set_title('Original Image')
	axes[1].imshow(transformed_image)
	axes[1].set_title('Transformed Image')
	results_dir = os.path.join('builder_scripts', 'Results')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	plt.show()
	plt.savefig(os.path.join(results_dir, 'crop_test.png'))
	# val_loader = vqa_dataset.get_loader(config, split='val')
	# print(val_loader.__getitem__(0))

