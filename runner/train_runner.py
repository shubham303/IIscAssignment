import torch
import wandb

from utils.save_best import SaveBestModel
from utils.utils import save_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def validate(val_dataloader, net, optimizer, criterion):
	val_runningloss = 0
	val_runningacc = 0
	total = 0
	correct = 0
	net.eval()
	for i, data in enumerate(val_dataloader):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		# zero the parameter gradients
		optimizer.zero_grad()
		
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		val_runningloss += loss.item()
		
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		
		accuracy = 100 * correct // total
		val_runningacc += accuracy
	
	val_runningacc = val_runningacc / len(val_dataloader)
	val_runningloss = val_runningloss / len(val_dataloader)
	
	return val_runningloss, val_runningacc


def train_model(train_loader, val_dataloader, epoch, net, optimizer, criterion, save_best):
	running_loss = 0.0
	
	batch_size = train_loader.batch_size
	log_iteration = 2000 // batch_size
	val_ratio = 8000 // batch_size
	
	for i, data in enumerate(train_loader):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		# zero the parameter gradients
		optimizer.zero_grad()
		
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
		# print statistics
		running_loss += loss.item()
		
		if i % val_ratio == val_ratio - 1:
			val_loss, val_acc = validate(val_dataloader, net, optimizer, criterion)
			metric = {"val_loss": val_loss, "val_acc": val_acc}
			wandb.log(metric)
			if save_best(val_loss, val_acc, epoch, net, optimizer) == -1:
				return -1
		
		if i % log_iteration == log_iteration - 1:
			metric = {"train_loss": running_loss / log_iteration}
			wandb.log(metric)
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_iteration:.3f}')
			running_loss = 0.0
	return 0


def train_runner(num_epoch, train_loader, net, val_dataloader, optimizer, criterion, basename, image_size):
	save_best = SaveBestModel(criterion, image_size, basename)
	
	for epoch in range(num_epoch):  # loop over the dataset multiple times
		val_loss, val_acc =validate(val_dataloader, net, optimizer, criterion)
		metric = {"val_loss": val_loss, "val_acc": val_acc}
		print(metric)
		if train_model(train_loader, val_dataloader, epoch, net, optimizer, criterion, save_best) == -1:
			#if model loss has not derceased for few iterations then early stop
			break
		
		filename = basename.replace(".pth", "_{}.pth".format(epoch))
		
		save_model(net, epoch, image_size, optimizer, criterion, filename)
	
	print('Finished Training')



