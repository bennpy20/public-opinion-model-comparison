"""
utils_embedding.py
Fungsi reusable untuk SKEMA 2 "Tokenizer": tokenizer subword IndoBERT + embedding layer IndoBERT
pralatih yang DIBEKUKAN (requires_grad=False)

Dipakai di: Part 04b, Part 05b, Part 06b

Yang diambil dari "indobenchmark/indobert-base-p1" hanya:
(1) tokenizer subword (WordPiece)
(2) tabel embedding token (word_embeddings, 768-dim)

Encoder transformer-nya tidak dipakai dan dibuang dari memory
Tidak ada fine-tuning
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "indobenchmark/indobert-base-p1"
PRETRAINED_DIM = 768


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer_and_frozen_embedding(model_name=MODEL_NAME):
    """
    Return (tokenizer, frozen_embedding)
    frozen_embedding: nn.Embedding (vocab_subword x 768), bobot disalin dari IndoBERT, frozen
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    weight = bert.get_input_embeddings().weight.detach().clone() # tabel embedding token pralatih
    del bert # encoder tidak dipakai -> buang
    emb = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=tokenizer.pad_token_id)
    emb.weight.requires_grad = False
    return tokenizer, emb


def encode_subword(texts, tokenizer, max_len=128):
    """
    Tokenisasi subword IndoBERT tanpa token khusus ([CLS]/[SEP]), dipotong/di-pad ke max_len
    Return: input_ids LongTensor (N, max_len), mask BoolTensor (N, max_len) True = token asli
    """
    enc = tokenizer(list(texts), add_special_tokens=False, truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"]
    mask = ids != tokenizer.pad_token_id
    return ids, mask


@torch.no_grad()
def mean_pool_embeddings(input_ids, mask, embedding, device=None, batch_size=512):
    """
    Rata-rata vektor embedding (beku) dari token asli tiap teks -> np.ndarray (N, 768)
    Teks tanpa token sama sekali -> vektor nol
    """
    device = device or next(embedding.parameters()).device
    embedding = embedding.to(device).eval()
    out = []
    for s in range(0, len(input_ids), batch_size):
        ids = input_ids[s:s + batch_size].to(device)
        m = mask[s:s + batch_size].to(device).unsqueeze(-1).float()
        e = embedding(ids) * m # padding dinolkan
        out.append((e.sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.vstack(out).astype(np.float32)
