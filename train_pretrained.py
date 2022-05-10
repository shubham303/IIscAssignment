import os

import torch

import wandb
from train import train
from utils.utils import pretrained_model, load_best_acc_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


default_config = {
		"lr": 0.001,
		"weight_decay": 0.00001,
		"epochs": 10,
		"batch_size": 64,
		"optimizer": "adam",
		"image_size": 224,
		"new_layer" : True,
		"train_layer" : 10,
		"finetune_lr" : 0.0                   # if finetune lr > 0 then finetune whole model with very small
		# learning rate
	}

# 1. Start a W&B run
wandb.init(config=default_config, project='cifar-10-finetuning')
config = wandb.config
net = pretrained_model(config["new_layer"], config["train_layer"]).to(device)
os.makedirs("/media/shubham/One Touch/iisc_asssignment/finetuned/{}/".format(wandb.run.id))
path = "/media/shubham/One Touch/iisc_asssignment/finetuned/{}/cifar_net.pth".format(wandb.run.id)
train(net, config, path)

if config["finetune_lr"] != 0:
	net = load_best_acc_model(path).to(device)
	run_id = wandb.run.id
	train(net, config, path)

torch.save(net.state_dict(), path)

