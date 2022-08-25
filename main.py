import logging
import os
import utils
from utils.downloadWord2vec import download_word2vec_newspaper
from utils.prepareData import _return_stopword_removed_dataframe, _train_test_split
from utils.weightedTfIdfVectorizer import TfIdfEmbeddingVectorizer
from torch.utils.data import DataLoader

from model.siameseDataset import SiameseDataset
from model.siameseNetwork import SiameseNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('input_size', type=int)
	parser.add_argument('hidden_size', type=int)
	parser.add_argument('num_layers', type=int)
	parser.add_argument('dropout', type=float)
	parser.add_argument('num_classes', type=int)
	parser.add_argument('bidirectional', type=bool)
	parser.add_argument('batch_size', type=int)
	parser.add_argument('epoch', type=int)

	args = parser.parse_args()


def train(args):
	word2vec = download_word2vec_newspaper()
	df = _return_stopword_removed_dataframe
	train_df, test_df = _train_test_split(df)
	documents = df['question1_without_stopword'].tolist() + df['question2_without_stopword'].tolist()

	tfidfVectorizer = TfIdfEmbeddingVectorizer()
	tfidfVectorizer.fit(documents)
	train_dataset = SiameseDataset(train_df, tfidfVectorizer)
	test_dataset = SiameseDataset(test_df, tfidfVectorizer)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


	model = SiameseNetwork(
		input_size=args.input_size,
		hidden_size=args.hidden_size,
		num_layers=args.num_layers,
		dropout=args.dropout,
		num_classes=args.num_classes,
		bidirectional=args.bidirectional
	)


	loss_fn = torch.nn.NLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)
	


	for epoch in range(args.epoch):
		cur_loss = 0.0
		val_loss = 0.0
		cur_acc = 0.0
		total = 0
		model.train()
		for idx, data in enumerate(train_dataloader):
			q1, q2, label = data
			optimizer.zero_grad()
			outputs = model(q1, q2)
			loss = loss_fn(outputs, label)
			loss.backward()
			optimizer.step()
			cur_loss += loss.itm()
		
		with torch.no_grad():
			model.eval()
			for _, data in enumerate(test_dataloader):
				q1, q2, label = data
				outputs = model(q1, q2)
				loss = loss_fn(outputs, label)
				_, pred = torch.max(outputs, 1)
				val_loss += loss.item()
				cur_acc += (pred == label).sum().item()
				total += outputs.size(0)

		acc = cur_acc * 100 / total
		print('Epoch {}:\t Acc {}'.format(epoch, acc))


if __name__ == '__main__':
	main()





