import torch

from configuration import classes
from models.model import Resnet
from runner.test_runner import test_runner
from utils.dataloader import get_testloader
from utils.utils import get_transform, pretrained_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(path):
    batch_size = 64
    import time
    start_time = time.time()
    net = Resnet(len(classes), False).to(device)
    #net = pretrained_model(True, 30).to(device)
    net_dict = torch.load(path)
    net.load_state_dict(net_dict["model_state_dict"])
    image_size = net_dict["image_size"]
    transform = get_transform(image_size)
    testloader = get_testloader(batch_size, transform)
    test_runner(net, testloader)
    print(time.time()-start_time)
if __name__ == "__main__":
    test("/media/shubham/One Touch/iisc_asssignment/zero_pruning/5tufi56z/cifar_net_best_acc.pth")