import os
import random


from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.misc

import sys

class pokemondataloader(data.Dataset):
	def __init__(self,image_dir):
		self.filename = [os.path.join(image_dir,i) for i in sorted(os.listdir(image_dir))]


	def __getitem__(self,i):
		png = Image.open(self.filename[i]).convert('RGBA')
		# img = Image.fromarray(img)
		img = Image.new("RGB", png.size, (255, 255, 255))
		img.paste(png, mask=png.split()[3])
		img = transforms.CenterCrop(64)(img)	
		img = transforms.ToTensor()(img)
		img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
		return img

	def __len__(self):
		return len(self.filename)


