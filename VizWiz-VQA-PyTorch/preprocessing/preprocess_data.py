import numpy as np
import os
from PIL import Image
from pdb import set_trace as bp
import torchvision.transforms as transforms

img_dir = '/home/ubuntu/data_vqa/Images/train'

mean_L = []
std_L = []
	
list_files = os.listdir(img_dir)
for i, file_batch in enumerate(np.array_split(np.random.permutation(list_files), 50)):
	print(i, len(file_batch))
	for j, file in enumerate(file_batch):
		img_path = os.path.join(img_dir, file)
		img = Image.open(img_path).convert('RGB')
		img = np.array(img).reshape(-1,3)

		img = transforms.ToTensor()(img).numpy()[0,:,:]

		mean = np.mean(img, axis=0).reshape(-1,1)
		std = np.std(img, axis=0).reshape(-1,1)

		if len(mean_L) == 0: mean_L = mean
		else: mean_L = np.append(mean_L, mean, axis=1)

		if len(std_L) == 0: std_L = std
		else: std_L = np.append(std_L, std, axis=1)

print(np.mean(mean_L, axis=1), np.mean(std_L, axis=1))

# transforms.Normalize(mean=[120.13510175, 114.11419162, 104.03966103],
# 					std=[62.18012176, 60.93507182, 61.69753557]),

# [0.4711275  0.44754702 0.40802705] [0.24383631 0.23895378 0.24194406]
