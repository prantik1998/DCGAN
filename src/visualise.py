import os
import torch 
import torchvision.utils as utils

from PIL import Image

from .model import Generator

def visualise(config):
	noise = torch.randn(config["generateimages"],100, 1, 1, device=config["device"])
	generator = Generator().to(config["device"])
	generator.load_state_dict(torch.load(config["generator"]))
	output = generator(noise).detach().cpu()
	for j in range(output.size(0)):
		# img = transforms.ToPILImage()(output[j])
		# img = img.resize((128,128),Image.BICUBIC)
		# print(img.size)
		imgpath = f'{config["results"]}/result_{j+1}.png'
		utils.save_image(output[j],imgpath,normalize=True)
		img = Image.open(imgpath).convert("RGB")
		img = img.resize((128,128),Image.BICUBIC)
		img.save(imgpath)