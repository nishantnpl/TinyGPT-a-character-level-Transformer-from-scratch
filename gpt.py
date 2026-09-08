from tokenizer import CharTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

#text = open('data.txt', encoding='utf-8').read()

text = open('data.txt', encoding='utf-8').read()
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # whole corpus as one int tensor
n = int(0.9 * len(data))
train_data, val_data =data[:n], data[n:] # train validation split

#Batching Function
block_size = 8
batch_size = 4
n_embd = 64

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

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                    # (B, T, 64)
        pos_emb = self.position_embedding_table(torch.arange(T))     # (T, 64)
        x = tok_emb + pos_emb                                        # (B, T, 64)
        logits = self.lm_head(x)                                     # (B, T, 50)
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


model = GPTModel()
optimizer = torch.optim.AdamW(model.parameters(), lr= 3e-3)

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
print(tokenizer.decode(idx[0].tolist()))
