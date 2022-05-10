import torch

from hooks.weightfreezehook import register_weigh_freeze_hook
from models.utils import pruning_cut_off, get_parameters_list, count_zero_weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_zero_pruned_model(model, prune_percent):
	# collect all weights in one list
	parameters_to_prune=get_parameters_list(model, [torch.nn.Linear, torch.nn.Conv2d], ["weight"])
	total_weight_parameters , cutoff = pruning_cut_off(parameters_to_prune, prune_percent)
	
	# make weights whose l1 norm is less than cutoff to zero
	for p in parameters_to_prune:
		p.data = torch.where(torch.abs(p) > cutoff, p, torch.tensor(0, dtype=p.dtype).to(device))
	
	zero_weights = count_zero_weights(model)
	size_ratio = 100 * ((total_weight_parameters - zero_weights) / total_weight_parameters)
	
	print("total_weights: {} zero_weight : {} ratio: {}".format(total_weight_parameters, zero_weights,size_ratio))
	for p in parameters_to_prune:
		register_weigh_freeze_hook(p, cutoff)
		
	return model, cutoff


def get_freezed_model(model, freeze_percent):
	parameters_to_freeze = get_parameters_list(model, [torch.nn.Linear, torch.nn.Conv2d], ["weight"])
	total_weight_parameters, cutoff = pruning_cut_off(parameters_to_freeze, freeze_percent)
	for p in parameters_to_freeze:
		register_weigh_freeze_hook(p, cutoff)
	return model, cutoff


	
