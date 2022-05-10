import torch


def pruning_cut_off(parameters_to_prune, prune_percent):
	total_weight_parameters =0
	# count  number of parameters
	for p in parameters_to_prune:
		total_weight_parameters += p.numel()
	
	# find index of cutoff weight value
	index_of_cut_off_weight = int(prune_percent * total_weight_parameters + 1)
	
	# find cutoff value
	temp = None
	for p in parameters_to_prune:
		a = p.data.view(-1)
		if temp is None:
			temp = a
		else:
			temp = torch.cat((temp, a), dim=0)
	
	cutoff = torch.kthvalue(abs(temp), index_of_cut_off_weight).values.item()
	
	return total_weight_parameters , cutoff


def get_parameters_list(model,module_type,parameter_type):
	parameters = []
	for name, module in model.named_modules():
		if type(module) in module_type:
			for p_name, p in module.named_parameters():
				if p_name in parameter_type:
					parameters.append(p)
					
	return parameters


def count_zero_weights(model):
	# for debugging purpose to check if number of zero weights are same before and after training.
	zero_weights = 0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
			for p_name, p in module.named_parameters():
				if "weight" in p_name:
					zero_weights += torch.sum(p == 0)
	return zero_weights