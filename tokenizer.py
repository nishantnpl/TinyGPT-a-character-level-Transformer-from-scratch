class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

if __name__ == '__main__':
    t = CharTokenizer(open('data.txt', encoding='utf-8').read())
    assert t.decode(t.encode("Alice")) == "Alice"
    print(t.vocab_size, t.encode("Alice"))