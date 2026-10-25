"""Tiny VLM: ViT image encoder -> linear projector -> causal transformer decoder (prefix-LM).

Attention is written with plain matmuls so the ONNX graphs run in onnxruntime-web.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, bias=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias
        y = att.softmax(-1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias=None):
        x = x + self.att(self.n1(x), bias)
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    def __init__(self, img=96, patch=12, d=160, layers=4, heads=4):
        super().__init__()
        self.stem = nn.Sequential(  # small conv stem keeps thin lines visible to the patches
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, d, patch, stride=patch))
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class TinyVLM(nn.Module):
    def __init__(self, vocab, d=160, enc_layers=4, dec_layers=4, heads=4, max_len=96):
        super().__init__()
        self.encoder = VisionEncoder(d=d, layers=enc_layers, heads=heads)
        self.project = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(dec_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def encode_image(self, img):
        return self.project(self.encoder(img))

    def decode(self, img_tokens, ids):
        """img_tokens (B, N, D), ids (B, T) -> logits (B, T, V). Image tokens attend
        bidirectionally among themselves; text tokens attend causally."""
        N, T = img_tokens.shape[1], ids.shape[1]
        x = torch.cat([img_tokens, self.tok(ids) + self.pos[:, :T]], 1)
        L = N + T
        r = torch.arange(L, device=ids.device)
        allowed = (r[None, :] <= r[:, None]) | (r[None, :] < N)
        bias = torch.zeros(L, L, device=ids.device).masked_fill(~allowed, -1e4)
        for b in self.blocks:
            x = b(x, bias)
        return self.head(self.norm(x[:, N:]))

    def forward(self, img, ids):
        return self.decode(self.encode_image(img), ids)

    @torch.no_grad()
    def generate(self, img, prompt_ids, eos, max_new=60):
        mem = self.encode_image(img)
        ids = prompt_ids
        for _ in range(max_new):
            nxt = self.decode(mem, ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if (nxt == eos).all():
                break
        return ids[:, prompt_ids.shape[1]:]
