import torch
from torch_geometric.datasets import UPFD
import random
import numpy as np
import matplotlib.pyplot as plt

SEED = 42

def set_seed():
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	return

def compare_bar_chart_with_std(array1, array2, std1, std2, label1='Fake', label2='Real', title='Feature Comparison'):
	array1 = np.array(array1)
	array2 = np.array(array2)
	std1 = np.array(std1)
	std2 = np.array(std2)

	if array1.shape != array2.shape or std1.shape != std2.shape or array1.shape != std1.shape:
		raise ValueError("Arrays and standard deviation arrays must have the same shape.")

	indices = np.arange(len(array1))
	bar_width = 0.35

	plt.figure(figsize=(10, 6))
		
	# Bar plots
	bar1 = plt.bar(indices - bar_width/2, array1, width=bar_width, label=label1, yerr=std1, capsize=5, alpha=0.7)
	bar2 = plt.bar(indices + bar_width/2, array2, width=bar_width, label=label2, yerr=std2, capsize=5, alpha=0.7)

	plt.xlabel('Feature Index')
	plt.ylabel('Value')
	plt.title(title)
	feature_names=['Verified','Enable geo-spatial positioning','Followers count','Friends count','Status count','Favorite count','Number of lists','Created time','Number of words in the description','Number of words in the screen name']
	plt.xticks(indices, feature_names, rotation=80)
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.show()


def main():

	train_dataset = UPFD(root='data/', name='politifact', feature='profile', split='train')
	val_dataset = UPFD(root='data/', name='politifact', feature='profile', split='val')
	test_dataset = UPFD(root='data/', name='politifact', feature='profile', split='test')

	g = torch.Generator()
	g.manual_seed(SEED)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	fake = []
	real = []
	for split in [test_dataset,train_dataset,val_dataset]:
		print("Got here")
		for data in split:
			print("1")
			data = data.to(device)
			for i in range(data.y.size(0)):
				if data.y[i].item() == 1:
					real.append(data.x[i])
				else:
					fake.append(data.x[i])
	
	print(np.array(fake).shape)
	print(np.array(real).shape)

	print(np.mean(fake,axis=0))
	print(np.mean(real,axis=0))

	compare_bar_chart_with_std(np.mean(fake,axis=0),np.mean(real,axis=0),np.std(fake,axis=0),np.std(real,axis=0))

if __name__=='__main__':
	set_seed()

	main()