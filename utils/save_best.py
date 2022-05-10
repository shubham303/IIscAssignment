from utils.utils import save_model


class SaveBestModel:
	def __init__(
			self,
			criterion, image_size, basename,
			best_valid_loss=float('inf'),
			best_valid_acc = 0,
			
	):
		self.best_valid_loss = best_valid_loss
		self.best_valid_acc = best_valid_acc
		self.criterion = criterion
		self.image_size = image_size
		self.basename = basename
		self.trigger = 0
		self.patience = 10
	
	def __call__(
			self, current_valid_loss,current_valid_acc,
			epoch, model, optimizer
	):
		if current_valid_loss < self.best_valid_loss:
			self.best_valid_loss = current_valid_loss
			print(f"\nBest validation loss: {self.best_valid_loss}")
			filename = self.basename.replace(".pth", "_best_loss.pth")
			save_model(model, epoch, self.image_size,optimizer, self.criterion , filename)
			self.trigger =0
		else:
			self.trigger+=1
			if self.trigger > self.patience:
				return -1
			
		if current_valid_acc  > self.best_valid_acc:
			self.best_valid_acc = current_valid_acc
			print(f"\nBest validation acc: {self.best_valid_acc}")
			filename = self.basename.replace(".pth", "_best_acc.pth")
			save_model(model, epoch, self.image_size,optimizer, self.criterion , filename)
		
		return 0