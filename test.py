from DatasetClasses import YelpDataset
from torch.utils.data import DataLoader
from gensim.models.keyedvectors import Word2VecKeyedVectors
from DataLoader_fns import save_vocab
import numpy as np
import torch
from Models import EncoderRNN, BinaryClassifier
from TrainModel import train_model
from DataLoader_fns import collate
from Inference_fns import get_accuracy

dataset_train = YelpDataset('dataset_train.json')
dataset_dev = YelpDataset('dataset_dev.json')
dataset_test = YelpDataset('dataset_test.json')

vocab = dataset_train.get_vocab()
save_vocab(vocab, 'vocab')