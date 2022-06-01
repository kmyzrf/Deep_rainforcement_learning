import torch.nn as nn	# access to layers
import torch.nn.functional as F	# access to sigmoid
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
	def __int__(self, lr, n_classes, input_dims):
		super(LinearClassifier, self).__init__()

		# 3 fuly connected layer
		self.fc1 = nn.Linear(*input_dims, 128)
		self.fc2 = nn.Linear(128, 256)
		self.fc3 = nn.Linear(256, n_classes)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.CrossEntropyLoss() # nn.MSELoss
		self.decive = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	# make activation layers
	def forward(self, data):
		layer1 = F.sigmoid(self.fc1(data))
		layer2 = F.sigmoid(self.fc2(layer1))
		layer3 = self.fc3(layer2)

		return layer3

	# make a larning loop
	def learn(self, data, labels):
		self.optimizer.zero_grad()
		data = T.tensor(data).to(self.device)
		labels = T.tensor(labels).to(self.device)
		
		predictions = self.forward(data)

		cost = self.loss(predictions, labels)

		cost.backward()
		self.optimizer.step()
		
