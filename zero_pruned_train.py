import os

import torch
from torch import nn

import configuration
import wandb
from models.model import Resnet
from models.pruned_models import get_zero_pruned_model
from models.utils import count_zero_weights
from runner.train_runner import train_runner
from utils.dataloader import get_trainvalloader, get_testloader
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
		optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
	else:
		optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
		                            momentum=0.9)
	
	transform = get_transform(config["image_size"])
	train_dataloader, _ = get_trainvalloader(config["batch_size"], transform)
	val_dataloader = get_testloader(config["batch_size"], transform)
	
	train_runner(config["epochs"], train_dataloader, net, val_dataloader, optimizer, criterion, save_path,
	             config["image_size"])
	
	torch.save(net.state_dict(), save_path)
if __name__ == "__main__":
	
	default_config = {
		"lr": 0.01,
		"weight_decay": 0.0001,
		"epochs": 1,
		"batch_size": 64,
		"optimizer": "sgd",
		"image_size": 224,
		"prune_percent": 0.5,
		"checkpoint" : "cifar_net_best_acc.pth"
	}
	
	# 1. Start a W&B run
	wandb.init(config=default_config, project='cifar-10-zero-pruning')
	config = wandb.config
	os.makedirs("/media/shubham/One Touch/iisc_asssignment/zero_pruning/{}/".format(wandb.run.id))
	path = "/media/shubham/One Touch/iisc_asssignment/zero_pruning/{}/cifar_net.pth".format(wandb.run.id)
	
	#create model
	net = Resnet(len(configuration.classes), 10).to(device)
	net_dict = torch.load(config["checkpoint"])
	net.load_state_dict(net_dict["model_state_dict"])
	#prune the model
	net, cutoff = get_zero_pruned_model(net, config["prune_percent"])
	
	x = count_zero_weights(net)
	train(net, config, path)
	print(x , "   ", count_zero_weights(net))  # ensure number of zero weights is same before and after training

