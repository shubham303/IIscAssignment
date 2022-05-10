import torch
import torchvision


def get_trainvalloader(batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                            download=True, transform=transform)
    
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    
    valloader=  torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    
    return trainloader, valloader

def get_testloader(batch_size, transform):
  
    
    testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    
    return  testloader

