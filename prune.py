import torch

import wandb
from configuration import classes
from models.model import Resnet
from models.pruned_models import get_zero_pruned_model
from utils.dataloader import get_testloader
from utils.utils import get_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PATH = "cifar_net_best_acc.pth"
batch_size = 64
image_size = 224

transform = get_transform(image_size)
testloader = get_testloader(batch_size, transform)

wandb.init(project="cifar-10-prune-0")

prune_percent  =0
while prune_percent <=0.9:
	model = Resnet(len(classes), False).to(device)
	net_dict = torch.load(PATH)
	model.load_state_dict(net_dict["model_state_dict"])
	model, _ =  get_zero_pruned_model(model, prune_percent)

	model.eval()
	correct = 0
	total = 0
	
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			# calculate outputs by running images through the network
			outputs = model(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	
	accuracy = 100 * correct // total
	
	metric = {"accuracy" : accuracy , "prune_percent" : prune_percent*100}
	print(metric)
	wandb.log(metric)
	prune_percent+=0.05