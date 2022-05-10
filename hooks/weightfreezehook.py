import torch


class weightFreezeHook:
	def __init__(self, w, cutoff):
		self.mask = (torch.abs(w) > cutoff).long()
	def __call__(self, grad):
		grad = grad * self.mask
		return grad
	
	def clear(self):
		self.mask = None


def register_weigh_freeze_hook(parameter,cutoff):
	# set all weights< cutoff to zero. apply weight freeze hook
	hook = weightFreezeHook(parameter, cutoff)
	parameter.register_hook(hook)
