import torch
import math
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.nn import global_mean_pool,GCNConv
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Dataset
import umap.umap_ as umap
from explain_model import explain_model

SEED = 42

def set_seed():
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	return

class TopIndicesDataset(Dataset):
    def __init__(self, original_dataset, top_indices):
        self.dataset = original_dataset
        self.top_indices = top_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        data.x = data.x[:, self.top_indices]
        return data

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


	def forward(self, x, edge_index, batch):
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
		
class GCN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super().__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.lin = torch.nn.Linear(hidden_channels, out_channels)

	def forward(self, x, edge_index, batch):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		x = global_mean_pool(x, batch)  # Pool node features into graph-level representation
		return self.lin(x)	

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

	def plot_embeddings(embeddings, labels, method="tsne"):
		if method == "tsne":
			reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=SEED)
		elif method == "umap":
			reducer = umap.UMAP(n_components=2,random_state=SEED)
		else:
			raise ValueError("Unknown method")

		emb_2d = reducer.fit_transform(embeddings)

		plt.figure(figsize=(13, 7))
		scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, marker='o')
		plt.legend(*scatter.legend_elements(), title="Classes")
		plt.title(f"{method.upper()} visualization of embeddings")
		plt.show()
	
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

	architecture = 'GraphSAGE'
	feature = 'spacy'
	batch_size = 21
	num_layers = 3
	hidden_channels = 118
	learning_rate = 0.0073
	weight_decay = 0.00034

	train_dataset = UPFD(root='data/', name='gossipcop', feature=feature, split='train')
	val_dataset = UPFD(root='data/', name='gossipcop', feature=feature, split='val')
	test_dataset = UPFD(root='data/', name='gossipcop', feature=feature, split='test')

	g = torch.Generator()
	g.manual_seed(SEED)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, generator=g)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, generator=g)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	early_stopping = EarlyStopping(patience=15, delta=0.0005)

	if architecture == 'GraphSAGE':

		def extract_embeddings_from_path(model, loader, device, checkpoint_path, layer="last_hidden"):
			model = GraphSAGE(
				dataset=train_dataset,
				num_layers=num_layers,  # must match your config
				hidden=hidden_channels      # same here
			).to(device)

			model.load_state_dict(torch.load(checkpoint_path))
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

		model = GraphSAGE(
			dataset=train_dataset,		 # Needed for num_features and num_classes
			num_layers=num_layers,				  # You can tune this (e.g., 2 or 3)
			hidden=hidden_channels					  # Matches config
		).to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		criterion = torch.nn.CrossEntropyLoss()
		
		def train():
			model.train()
			total_loss = 0
			for data in train_loader:
				data = data.to(device)
				optimizer.zero_grad()
				out = model(data.x, data.edge_index, data.batch)
				loss = criterion(out, data.y)
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * data.num_graphs
			return total_loss / len(train_loader.dataset)
		
		def evaluate(loader):
			model.eval()
			all_preds = []
			all_labels = []
			with torch.no_grad():
				for data in loader:
					data = data.to(device)
					out = model(data.x, data.edge_index, data.batch)
					pred = out.argmax(dim=1)
					all_preds.extend(pred.cpu().numpy())
					all_labels.extend(data.y.cpu().numpy())

			acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
			f1 = f1_score(all_labels, all_preds, average='binary')
			return acc, f1

		list_paths = []
		for epoch in range(500):
			loss = train()
			val_acc, val_f1 = evaluate(val_loader)
			test_acc, test_f1 = evaluate(test_loader)
			if epoch%10==0:
				list_paths.append(f'model_epoch_{epoch}.pt')
				torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
			# 	plot_embeddings(embeddings.numpy(), labels.numpy(), method="umap")
			print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
			early_stopping(loss, model)
			if early_stopping.early_stop:
				print("Early stopping")
				break

		for path in list_paths:
			embeddings_end, labels_end = extract_embeddings_from_path(model, test_loader, device, path, layer="last_hidden")
			plot_embeddings(embeddings_end.numpy(), labels_end.numpy(), method="umap")
		
		embeddings_end, labels_end = extract_embeddings(model, test_loader, device, layer="last_hidden")
		plot_embeddings(embeddings_end.numpy(), labels_end.numpy(), method="umap")
		
	elif architecture == 'GCN':
			
		# Instantiate model, optimizer, and define training
		model = GCN(
			in_channels=train_dataset.num_node_features,
			hidden_channels=hidden_channels,
			out_channels=2  # Binary classification (real/fake)
		).to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		criterion = torch.nn.CrossEntropyLoss()

		def train():
			model.train()
			total_loss = 0
			for data in train_loader:
				data = data.to(device)
				optimizer.zero_grad()
				out = model(data.x, data.edge_index, data.batch)
				loss = criterion(out, data.y)
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * data.num_graphs
			return total_loss / len(train_loader.dataset)

		def evaluate(loader):
			model.eval()
			all_preds = []
			all_labels = []
			with torch.no_grad():
				for data in loader:
					data = data.to(device)
					out = model(data.x, data.edge_index, data.batch)
					pred = out.argmax(dim=1)
					all_preds.extend(pred.cpu().numpy())
					all_labels.extend(data.y.cpu().numpy())

			acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
			f1 = f1_score(all_labels, all_preds, average='binary')
			return acc, f1

		for epoch in range(500):
			loss = train()
			val_acc, val_f1 = evaluate(val_loader)
			test_acc, test_f1 = evaluate(test_loader)
			print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
			early_stopping(loss, model)
			if early_stopping.early_stop:
				print("Early stopping")
				break


	top_indices = explain_model(
		model,
		architecture,
		test_dataset,
		device,
		feat_path=f'feature_importance_GraphSAGE.png',
		top_k=10,
		subgraph_path=f'subgraph_GraphSAGE.png'
	)

	# Create new datasets with only top features
	train_dataset_top = TopIndicesDataset(train_dataset, top_indices)
	val_dataset_top = TopIndicesDataset(val_dataset, top_indices)
	test_dataset_top = TopIndicesDataset(test_dataset, top_indices)

	# Create new DataLoaders
	train_loader_top = DataLoader(train_dataset_top, batch_size=batch_size, shuffle=True, generator=g)
	val_loader_top = DataLoader(val_dataset_top, batch_size=batch_size, generator=g)
	test_loader_top = DataLoader(test_dataset_top, batch_size=batch_size, generator=g)
	
	in_channels = len(top_indices)
	model_top = GraphSAGE(
		dataset=train_dataset_top, 
		num_layers=num_layers,
		hidden=hidden_channels
	).to(device)

	optimizer_top = torch.optim.Adam(model_top.parameters(), lr=learning_rate, weight_decay=weight_decay)
	criterion_top = torch.nn.CrossEntropyLoss()

	def train_top():
		model_top.train()
		total_loss = 0
		for data in train_loader_top:
			data = data.to(device)
			optimizer_top.zero_grad()
			out = model_top(data.x, data.edge_index, data.batch)
			loss = criterion_top(out, data.y)
			loss.backward()
			optimizer_top.step()
			total_loss += loss.item() * data.num_graphs
		return total_loss / len(train_loader_top.dataset)

	def evaluate_top(loader):
		model_top.eval()
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for data in loader:
				data = data.to(device)
				out = model_top(data.x, data.edge_index, data.batch)
				pred = out.argmax(dim=1)
				all_preds.extend(pred.cpu().numpy())
				all_labels.extend(data.y.cpu().numpy())
		acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
		f1 = f1_score(all_labels, all_preds, average='binary')
		return acc, f1

	# Training loop for the reduced model
	early_stopping_top = EarlyStopping(patience=15, delta=0.0005)
	for epoch in range(500):
		loss = train_top()
		val_acc, val_f1 = evaluate_top(val_loader_top)
		test_acc, test_f1 = evaluate_top(test_loader_top)
		print(f'[Top 5%] Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
		early_stopping_top(loss, model_top)
		if early_stopping_top.early_stop:
			print("[Top 5%] Early stopping")
			break

	return

if __name__=='__main__':
	set_seed()

	main()