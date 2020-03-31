import torch

import os

from .train import train 
from .visualise import visualise
from .dataset import pokemondataloader

import torchvision
import torchvision.transforms as transforms

class pipeline_manager:
	def __init__(self,config):
		self.config = config

	def visualise(self):
		visualise(self.config) 

	def train(self):
		dataset = pokemondataloader(self.config['dataset'])
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'],shuffle=True, num_workers=4)
		# transform = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
		# dataloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=2)		
		train(self.config,dataloader)