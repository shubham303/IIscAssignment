import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

import configuration
from models.model import ResnetPretrained

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def show_imgs(classes, images, labels, batch_size):

	# functions to show an image
	def imshow(img):
		img = img / 2 + 0.5  # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
	
	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	
def get_transform(size):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Resize(size = size),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	
	return transform


def save_model(model, epoch, image_size,optimizer, criterion , filename):
	torch.save({
		'epoch': epoch + 1,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	 	'loss': criterion,
		'image_size' : image_size,
	}, filename)


def pretrained_model(new_layer, train_layers):
	"""
	
	:param new_layer: if set to True, new FC layer is added on top of final FC layer
	:param train_layers: percentage of layers to finetune
	:return: return model
	"""
	model = ResnetPretrained(len(configuration.classes), new_layer)
	
	for p in list(model.parameters()):
		p.requires_grad = False
	
	
	finetune_layer_percentage = train_layers
	n = len(list(model.parameters()))
	start = n - int(n * (finetune_layer_percentage / 100))
	for i, p in enumerate(list(model.parameters())[start:]):
		p.requires_grad = True
	
	return model

def load_best_acc_model(basename):
	filename = basename.replace(".pth" ,"_best_acc.pth")
	return torch.load(filename)
	