from torch_geometric.datasets import UPFD
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import wandb

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

def main():

	wandb.init(
		project="gnn-fake-news",    # Your wandb project name
		name="gcn-upfd-run-1",      # Custom name (optional)
		config={
			"architecture": "GCN",
			"dataset": "UPFD - politifact",
			"features": "profile",
			"batch_size": 32,
			"hidden_channels": 64,
			"epochs": 100,
			"learning_rate": 0.1,
			"weight_decay": 5e-4,
		}
	)

	# Load dataset (Cora as example)
	train_dataset = UPFD(root='data/', name='politifact', feature='profile', split='train')
	val_dataset = UPFD(root='data/', name='politifact', feature='profile', split='val')
	test_dataset = UPFD(root='data/', name='politifact', feature='profile', split='test')

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32)
	test_loader = DataLoader(test_dataset, batch_size=32)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Instantiate model, optimizer, and define training
	model = GCN(
		in_channels=train_dataset.num_node_features,
		hidden_channels=64,
		out_channels=2  # Binary classification (real/fake)
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()

	# Training Loop
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
		correct = 0
		for data in loader:
			data = data.to(device)
			out = model(data.x, data.edge_index, data.batch)
			pred = out.argmax(dim=1)
			correct += int((pred == data.y).sum())
		return correct / len(loader.dataset)

	for epoch in range(1, 101):
		loss = train()
		val_acc = evaluate(val_loader)
		test_acc = evaluate(test_loader)
		wandb.log({
			"epoch": epoch,
			"loss": loss,
			"val_acc": val_acc,
			"test_acc": test_acc
		})
		wandb.watch(model, log="all", log_freq=10)
		print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

	wandb.finish()

	return

if __name__ == '__main__':
	main()