import os

import torch
from torch import nn

import configuration
import wandb
from models.model import Resnet
from runner.train_runner import train_runner
from utils.dataloader import get_trainvalloader
from utils.utils import get_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train(net, config, save_path="./"):
	
	"""
	:param checkpoint: resume trainging from checkpoint
	:param prune:
	:param freeze:
	:return:
	"""
	
	criterion = nn.CrossEntropyLoss()
	
	if config["optimizer"] == 'adam':
		optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay = config["weight_decay"])
	else:
		optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9)
	
	transform = get_transform(config["image_size"])
	train_dataloader, val_dataloader = get_trainvalloader(config["batch_size"], transform)
	
	train_runner(config["epochs"], train_dataloader,net,  val_dataloader, optimizer, criterion, save_path,
	             config["image_size"])
	
	torch.save(net.state_dict(), save_path)
	
if __name__ == "__main__":
	
	default_config = {
		"lr": 0.001,
		"weight_decay": 0.00001,
		"epochs": 10,
		"batch_size": 64,
		"optimizer": "adam",
		"image_size": 224,
	}
	
	# 1. Start a W&B run
	wandb.init(config=default_config, project='cifar-10')
	config = wandb.config
	os.makedirs("/media/shubham/One Touch/iisc_asssignment/{}/".format(wandb.run.id))
	path = "/media/shubham/One Touch/iisc_asssignment/{}/cifar_net.pth".format(wandb.run.id)
	
	net = Resnet(len(configuration.classes), 10).to(device)
	train(net,config,path)
	
	

