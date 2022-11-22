import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

from dataset import Sequence_StrategyDataset
import classification_model

dataset = Sequence_StrategyDataset('../csvfiles/221120/train/processed/processed.csv')

batch_size = 64

n_samples = len(dataset)
train_size = int(n_samples * 0.8)
val_size = n_samples - train_size - batch_size
test_size = batch_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, 64, shuffle=True)
val_loader = DataLoader(val_dataset, 64, shuffle=False)
test_loader = DataLoader(test_dataset, 64, shuffle=False)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = classification_model.Net().to(device)
print(f"Net : {model}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, dataloader, criterion, optimizer, device, epoch):
	print('Epoch {}/{}'.format(epoch+1, num_epochs))

	model.train()

	epoch_loss = 0.0
	epoch_corrects = 0

	for i, (inputs, labels) in enumerate(dataloader):
		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()
		output = model(inputs)
		loss = criterion(output, labels)
		_, preds = torch.max(output, 1)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()*inputs.size(0)
		epoch_corrects += torch.sum(preds == labels.data)

	epoch_loss = epoch_loss/len(dataloader.dataset)
	epoch_acc = epoch_corrects.double()/len(dataloader.dataset)

	print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def valid(model, dataloader, criterion, device, epoch):
	with torch.no_grad():
		epoch_loss = 0.0
		epoch_corrects = 0

		model.eval()
		for i , (inputs, labels) in enumerate(dataloader):
			inputs, labels = inputs.to(device), labels.to(device)

			output = model(inputs)
			loss = criterion(output, labels)

			_, preds = torch.max(output, 1)

			epoch_loss += loss.item()*inputs.size(0)
			epoch_corrects += torch.sum(preds == labels.data)

		epoch_loss = epoch_loss/len(dataloader.dataset)
		epoch_acc = epoch_corrects.double()/len(dataloader.dataset)

	print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

num_epochs = 100
for epoch in range(num_epochs):
	train(model, train_loader, criterion, optimizer, device, epoch=epoch)
	valid(model, val_loader, criterion, device, epoch=epoch)
	print("----------------------------------")

cpu = torch.device("cpu")
model.to(cpu)
torch.save(model.state_dict(), "imitation_model.pt")

def inference(input_data, label):
	import time
	testx = np.zeros((0, 230), dtype=np.float32)

	a = input_data
	a1 = np.expand_dims(a,axis=0)

	testx = np.append(testx, a1, axis=0)

	t0 = time.time()
	testy = model(torch.FloatTensor(testx))
	_, testy = torch.max(testy, 1)
	print(f"testy, label : {testy}, {label}")
	print('forward time [s]: ' + str(time.time()-t0))

model.eval()
batch_iterator = test_loader.__iter__()
#data, label = next(batch_iterator)
data, label = batch_iterator.next()
for i in range(data.shape[0]):
	inference(data[i], label[i])
