import torch 


config = {}

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config["lr"] = 1e-5
config["epochs"] = 100
config["checkpoint"] = "checkpoint"

config["dataset"] = "pokemons/images/images"
config["batch_size"] = 64
config["results"] = "results"

config["generator"] = "D:/DCGAN/checkpoint/generator_epoch_90.pth"
config["discriminator"] = "D:/DCGAN/checkpoint/discriminator_epoch_90.pth"

config["generateimages"] = 500
