import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

text = open('data.txt', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)} # char -> id
iots = {i: ch for i, ch in enumerate(chars)} # id -> char tokenized in both direction

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(iots[i]for i in l)

data = torch.tensor(encode, dtype=torch.long) # whole corpse as one int tensor
n = int(0.9 * len(data))
train_data, val_data =data[:n], data[n:]