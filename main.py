from config.config import config
from src.pipeline_manager import pipeline_manager

import click 

@click.group()
def main():
	pass

@main.command()
def train():
	manager.train()

@main.command()
def visualise():
	manager.visualise()


if __name__=="__main__":
	manager = pipeline_manager(config)
	main()


