import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.loader import DataLoader
import wandb
from sklearn.metrics import f1_score

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

def main():

	wandb.init(
		project="gnn-fake-news",	# Your wandb project name
		name="SAGE-upfd-run-1",	  # Custom name (optional)
		config={
			"architecture": "GraphSAGE",
			"dataset": "UPFD - politifact",
			"features": "bert",
			"batch_size": 32,
			"hidden_channels": 64,
			"num_layers": 2,
			"epochs": 100,
			"learning_rate": 0.001,
			"weight_decay": 5e-4,
		}
	)

	train_dataset = UPFD(root='data/', name='politifact', feature='bert', split='train')
	val_dataset = UPFD(root='data/', name='politifact', feature='bert', split='val')
	test_dataset = UPFD(root='data/', name='politifact', feature='bert', split='test')

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32)
	test_loader = DataLoader(test_dataset, batch_size=32)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = GraphSAGE(
		dataset=train_dataset,		 # Needed for num_features and num_classes
		num_layers=2,				  # You can tune this (e.g., 2 or 3)
		hidden=64					  # Matches config
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
		
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
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for data in loader:
				data = data.to(device)
				out = model(data)
				pred = out.argmax(dim=1)
				all_preds.extend(pred.cpu().numpy())
				all_labels.extend(data.y.cpu().numpy())

		acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
		f1 = f1_score(all_labels, all_preds, average='binary')  # Use 'macro' for multi-class
		return acc, f1

	for epoch in range(1, 101):
		loss = train()
		val_acc, val_f1 = evaluate(val_loader)
		test_acc, test_f1 = evaluate(test_loader)
		wandb.log({
			"epoch": epoch,
			"loss": loss,
			"val_acc": val_acc,
			"val_f1": val_f1,
			"test_acc": test_acc,
			"test_f1": test_f1
		})
		# wandb.watch(model, log="all", log_freq=10)
		print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

	wandb.finish()

	return

if __name__=='__main__':
	main()