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
# print(xb.shape)
# print(xb[0])
# print(yb[0])

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets= None):
        logits = self.token_embedding_table(idx)    # B, T -> (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

#Model creation. And took one batch . Forward pass without training, torch.Size([4, 8, 50]), and with the number we are checking
# model = BigramModel(vocab_size)
# xb, yb = get_batch('train')
# logits, loss = model(xb, yb)
# print(logits.shape)
# print(loss.item())


model = BigramModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr= 3e-3)

for step in range(3000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(step, loss.item())


idx = torch.zeros((1, 1), dtype=torch.long)      # start token
for _ in range(200):
    logits, _ = model(idx)
    probs = F.softmax(logits[:, -1, :], dim=-1)  # distribution over next char
    idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)  # sample, append
print(decode(idx[0].tolist()))
