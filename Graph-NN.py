import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create a graph representation of a simple computer network where A-> B-> C-> D-> E
G = nx.Graph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])

# Define the GNN architecture
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, adjacency_matrix):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.mm(adjacency_matrix, x)
        x = self.conv2(x)
        return x

# Define the input and target data
input_data = np.random.rand(5, 1) # random input feature for each node
target_data = np.random.rand(5, 1) # random target feature for each node
input_data = torch.from_numpy(input_data).float()
target_data = torch.from_numpy(target_data).float()

# Convert the graph to an adjacency matrix
adjacency_matrix = nx.to_numpy_matrix(G)
adjacency_matrix = torch.from_numpy(adjacency_matrix).float()

# Initialize the GNN
gnn = GNN(1, 8, 1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(gnn.parameters())

# Train the GNN
for epoch in range(100):
    output = gnn(input_data, adjacency_matrix)
    loss = criterion(output, target_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training complete')

#Combine the fuzzy and GNN
class FuzzyGNN(nn.Module):
def init(self, input_dim, hidden_dim, output_dim):
    super(FuzzyGNN, self).init()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.fuzzy = FuzzyLogic()

def forward(self, x, adjacency_matrix):
    x = self.conv1(x)
    x = torch.relu(x)
    x = torch.mm(adjacency_matrix, x)
    x = self.conv2(x)
    x = self.fuzzy(x) # apply fuzzy logic
    return x

#Test the new model on any new dummy network
dummy_network = nx.Graph()
dummy_network.add_nodes_from(['A', 'B', 'C'])
dummy_network.add_edges_from([('A', 'B'), ('B', 'C')])

dummy_input_data = np.random.rand(3, 1) # random input feature for each node
dummy_target_data = np.random.rand(3, 1) # random target feature for each node
dummy_input_data = torch.from_numpy(dummy_input_data).float()
dummy_target_data = torch.from_numpy(dummy_target_data).float()

dummy_adjacency_matrix = nx.to_numpy_matrix(dummy_network)
dummy_adjacency_matrix = torch.from_numpy(dummy_adjacency_matrix).float()

fuzzy_gnn = FuzzyGNN(1, 8, 1)

for epoch in range(100):
    output = fuzzy_gnn(dummy_input_data, dummy_adjacency_matrix)
    loss = criterion(output, dummy_target_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print('Testing complete')

