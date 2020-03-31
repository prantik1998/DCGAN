import torch
import torch.nn as nn

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(	 100, 64 * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(64 * 8),
			nn.ReLU(True),
			# state size. (64*8) x 4 x 4
			nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(True),
			# state size. (64*4) x 8 x 8
			nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 2),
			nn.ReLU(True),
			# state size. (64*2) x 16 x 16
			nn.ConvTranspose2d(64 * 2,	 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			# state size. (64) x 32 x 32
			nn.ConvTranspose2d(	64,3, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)

	def forward(self, input):
		# if input.is_cuda and self.ngpu > 1:
		# 	output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		# else:
		output = self.main(input)
		return output

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(3,64, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (64) x 32 x 32
			nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (64*2) x 16 x 16
			nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (64*4) x 8 x 8
			nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (64*8) x 4 x 4
			nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		output = self.main(input)
		return output.view(-1, 1).squeeze(1)


if __name__=="__main__":
	x = torch.randn(40,100,1,1)
	generator = Generator()
	print(generator(x).size())