import os 
import numpy as np

from .model import Generator,Discriminator

import torch
import torch.nn as nn
import torchvision.utils as utils
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

# import torchvision as vutils

def train(config,dataloader):
	generator = Generator().to(config['device'])
	discriminator = Discriminator().to(config['device'])
	if "generator" in config.keys():
		print("Loading Pretrained Model")
		generator.load_state_dict(torch.load(config["generator"]))
		discriminator.load_state_dict(torch.load(config["discriminator"]))
	else:
		print("Initialising Model")
	

	optimizerD = optim.Adam(discriminator.parameters(), lr=config["lr"])
	optimizerG = optim.Adam(generator.parameters(), lr=config["lr"])

	criterion = nn.BCELoss()

	# fixed_noise = torch.randn(config["batch_size"],100, 1, 1, device=config["device"])
	fixed_noise = torch.randn(1,100, 1, 1, device=config["device"])
	real_label = 1
	fake_label = 0

	for epoch in range(config["epochs"]):
		for i, data in enumerate(dataloader, 0):
			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			# train with real
			discriminator.zero_grad()
			real_cpu = data.to(config["device"])
			label = torch.full((real_cpu.size(0),), real_label, device=config["device"])
			
			output = discriminator(real_cpu)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(real_cpu.size(0),100, 1, 1, device=config["device"])
			fake = generator(noise)
			label.fill_(fake_label)  # fake labels are real for generator cost
			output = discriminator(fake.detach())
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			generator.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			output = discriminator(fake)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
				  % (epoch,config["epochs"], i, len(dataloader),
					 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

		if (epoch) % 10 == 0:
			fake = generator(fixed_noise)
			utils.save_image(fake.detach().cpu(),f"{config['results']}/img_{epoch}.png",normalize=True)
			# do checkpointing
			torch.save(generator.state_dict(), f'{config["checkpoint"]}/generator_epoch_{epoch}.pth')
			torch.save(discriminator.state_dict(), f'{config["checkpoint"]}/discriminator_epoch_{epoch}.pth')

