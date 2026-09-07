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

data = torch.tensor(encode(text), dtype=torch.long) # whole corpse as one int tensor
n = int(0.9 * len(data))
train_data, val_data =data[:n], data[n:]

#Batching Function
block_size = 8
batch_size = 4

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size]   for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(xb.shape)
print(xb[0])
print(yb[0])