import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_runner(net, testloader):
	net.eval()
	correct = 0
	total = 0
	
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			# calculate outputs by running images through the network
			outputs = net(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	
	print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
	return 100 * correct // total
