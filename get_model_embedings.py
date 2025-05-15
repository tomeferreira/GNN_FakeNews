import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import random
import numpy as np

SEED = 42

def set_seed():
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return

'''
Code from: https://wandb.ai/graph-neural-networks/GraphSAGE/reports/An-Introduction-to-GraphSAGE--Vmlldzo1MTEwNzQ1
'''
class GraphSAGE(torch.nn.Module):
	def __init__(self, dataset, num_layers, hidden):
		super().__init__()
		self.conv1 = SAGEConv(dataset.num_features, hidden)
		self.convs = torch.nn.ModuleList()
		for i in range(num_layers - 1):
			self.convs.append(SAGEConv(hidden, hidden))
		self.lin1 = Linear(hidden, hidden)
		self.lin2 = Linear(hidden, dataset.num_classes)


	def reset_parameters(self):
		self.conv1.reset_parameters()
		for conv in self.convs:
			conv.reset_parameters()
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = F.relu(self.conv1(x, edge_index))
		for conv in self.convs:
			x = F.relu(conv(x, edge_index))
		x = global_mean_pool(x, batch)
		x = F.relu(self.lin1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin2(x)
		return F.log_softmax(x, dim=-1)
	
	def get_embeddings(self, data, layer="last_hidden"):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = F.relu(self.conv1(x, edge_index))
		for conv in self.convs:
			x = F.relu(conv(x, edge_index))
		x = global_mean_pool(x, batch)

		if layer == "last_hidden":
			x = F.relu(self.lin1(x))  # optional: dropout if needed
			return x.detach().cpu(), data.y.cpu()
		elif layer == "before_pool":
			return x.detach().cpu(), data.y.cpu()

'''
code from: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
'''
class EarlyStopping:
	def __init__(self, patience=5, delta=0):
		self.patience = patience
		self.delta = delta
		self.best_score = None
		self.early_stop = False
		self.counter = 0
		self.best_model_state = None

	def __call__(self, val_loss, model):
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.best_model_state = model.state_dict()
		elif score < self.best_score + self.delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.best_model_state = model.state_dict()
			self.counter = 0
	def load_best_model(self, model):
		model.load_state_dict(self.best_model_state)

def main():

	train_dataset = UPFD(root='data/', name='politifact', feature='bert', split='train')
	val_dataset = UPFD(root='data/', name='politifact', feature='bert', split='val')
	test_dataset = UPFD(root='data/', name='politifact', feature='bert', split='test')

	g = torch.Generator()
	g.manual_seed(SEED)

	train_loader = DataLoader(train_dataset, batch_size=37, shuffle=True, generator=g)
	val_loader = DataLoader(val_dataset, batch_size=37, generator=g)
	test_loader = DataLoader(test_dataset, batch_size=37, generator=g)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = GraphSAGE(
		dataset=train_dataset,		 # Needed for num_features and num_classes
		num_layers=2,				  # You can tune this (e.g., 2 or 3)
		hidden=49					  # Matches config
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.006632297252690111, weight_decay=0.0005055585339535605)
	criterion = torch.nn.CrossEntropyLoss()
		
	early_stopping = EarlyStopping(patience=15, delta=0.0005)

	def train():
		model.train()
		total_loss = 0
		for data in train_loader:
			data = data.to(device)
			optimizer.zero_grad()
			out = model(data)
			loss = criterion(out, data.y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * data.num_graphs
		return total_loss / len(train_loader.dataset)
	 
	def evaluate(loader):
		model.eval()
		correct = 0
		for data in loader:
			data = data.to(device)
			out = model(data)
			pred = out.argmax(dim=1)
			correct += int((pred == data.y).sum())
		return correct / len(loader.dataset)

	def extract_embeddings(model, loader, device, layer="last_hidden"):
		model.eval()
		all_embeddings = []
		all_labels = []

		with torch.no_grad():
			for data in loader:
				data = data.to(device)
				embeddings, labels = model.get_embeddings(data, layer=layer)
				all_embeddings.append(embeddings)
				all_labels.append(labels)

		return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

	def plot_embeddings(embeddings, labels, method="tsne"):
		if method == "tsne":
			reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=SEED)
		elif method == "umap":
			reducer = umap.UMAP(n_components=2,random_state=SEED)
		else:
			raise ValueError("Unknown method")

		emb_2d = reducer.fit_transform(embeddings)

		plt.figure(figsize=(8, 6))
		scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
		plt.legend(*scatter.legend_elements(), title="Classes")
		plt.title(f"{method.upper()} visualization of embeddings")
		plt.show()

	for epoch in range(1, 101):
		loss = train()
		val_acc = evaluate(val_loader)
		test_acc = evaluate(test_loader)
		print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
		early_stopping(loss, model)
		if early_stopping.early_stop:
			print("Early stopping")
			break

	embeddings, labels = extract_embeddings(model, test_loader, device, layer="last_hidden")
	plot_embeddings(embeddings.numpy(), labels.numpy(), method="umap")

	return

if __name__=='__main__':
	set_seed()
	main()