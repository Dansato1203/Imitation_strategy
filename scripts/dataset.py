import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def make_sequence_data(data_frame, index, window):
	a = data_frame.iloc[index:index+window, :]
	return a

class Sequence_StrategyDataset(Dataset):
	def __init__(self, csv_path):
		df = pd.read_csv(csv_path)
		x = np.zeros((0, 230), dtype=np.float32)
		t = np.zeros((0), dtype=np.int32)

		for i in range(len(df)-5):
			d = make_sequence_data(df.drop('TASK', axis=1), i, 5)
			d = np.asarray(d).reshape(-1).astype(np.float32)
			#print(f"d : {d.shape}")
			d1 = np.expand_dims(d, axis=0)
			#print(f"d1 : {d1.shape}")
			x = np.append(x, d1, axis=0)
			print(f"x : {x.shape}")

			l = df.at[i+5-1, 'TASK']
			l = l.astype(np.int32)
			#print(f"label : {l}")
			l1 = np.expand_dims(l,axis=0)
			t = np.append(t, l1, axis=0)
			print(f"t : {t.shape}")

		#data = torch.tensor(df.drop('TASK', axis=1).values, dtype=torch.float64)
		#labels = torch.tensor(df['TASK'].values)

		self.data = torch.Tensor(x)
		self.label = torch.LongTensor(t)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()
		datas = self.data[index, :]
		labels = self.label[index]
		#print(f"datas, labels: {datas.shape}, {labels.shape}")
		return (datas, labels)

"""
dataset = Sequence_StrategyDataset('normalize_test.csv')
features_sample, label_sample = dataset[0]
print(features_sample, label_sample)

#train_loader = DataLoader(dataset, args.batch_size, shuffle=False)
train_loader = DataLoader(dataset, 8, shuffle=False)

batch_iterator = iter(train_loader)
inputs, labels = next(batch_iterator)
print(inputs)
print(labels[0])
print(inputs.size())
print(labels.size())
"""
