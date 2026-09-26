"""PREPARE for nanochat-lite: climbmix shards (karpathy/autoresearch data, MIT code; ideas/numbers only from
mazar/autoresearch-spark), a vocab-8192 BPE tokenizer, a uint16 training token stream and two disjoint fixed eval sets.

Validation and holdout come from the pinned upstream validation shard (never used for training): its documents are
split in half (first half → validation, second half → holdout), packed BOS-aligned into rows of SEQ_LEN+1 tokens.
"""
import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
TRAIN_SHARDS = list(range(8))
VAL_SHARD = 6542
VOCAB_SIZE = 8192
SEQ_LEN = 1024
TOKENIZER_CHARS = 400_000_000
MAX_TRAIN_TOKENS = 400_000_000
EVAL_ROWS = 2048                      # per split: 2048 × 1024 ≈ 2.1M scored tokens
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS = "<|reserved_0|>"
RAW = "/tmp/raw"


def fetch(i):
    path = f"{RAW}/shard_{i:05d}.parquet"
    for attempt in range(5):
        try:
            with requests.get(f"{BASE_URL}/shard_{i:05d}.parquet", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path + ".tmp", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".tmp", path)
            return path
        except (requests.RequestException, OSError) as e:
            print(f"retry {i}: {e}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: shard {i}")


def docs(path):
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        yield from pf.read_row_group(g).column("text").to_pylist()


def pack_rows(enc, bos, texts, n_rows):
    stream = []
    for i in range(0, len(texts), 256):
        for ids in enc.encode_ordinary_batch(texts[i:i + 256], num_threads=16):
            stream.append(bos)
            stream.extend(ids)
        if len(stream) >= n_rows * (SEQ_LEN + 1):
            break
    arr = np.asarray(stream[: n_rows * (SEQ_LEN + 1)], dtype=np.uint16)
    assert len(arr) == n_rows * (SEQ_LEN + 1), f"not enough eval text: {len(arr)}"
    return arr.reshape(n_rows, SEQ_LEN + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = ap.parse_args().out
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        train_paths = list(ex.map(fetch, TRAIN_SHARDS))
    val_path = fetch(VAL_SHARD)
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)

    def text_iter():
        n = 0
        for p in train_paths:
            for t in docs(p):
                t = t[:10_000]
                n += len(t)
                yield t
                if n >= TOKENIZER_CHARS:
                    return
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(text_iter(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    enc = tiktoken.Encoding(name="rustbpe", pat_str=tok.get_pattern(), mergeable_ranks=ranks,
                            special_tokens={n: len(ranks) + i for i, n in enumerate(SPECIAL_TOKENS)})
    bos = enc.encode_single_token(BOS)
    token_bytes = np.array([0 if enc.decode([i]) in SPECIAL_TOKENS else len(enc.decode_single_token_bytes(i))
                            for i in range(enc.n_vocab)], dtype=np.int32)
    print(f"tokenizer trained in {time.time() - t0:.0f}s (vocab {enc.n_vocab})", flush=True)

    os.makedirs(f"{out}/train")
    with open(f"{out}/train/tokens.bin", "wb") as f:
        n = 0
        for p in train_paths:
            texts = list(docs(p))
            for i in range(0, len(texts), 1024):
                chunk = []
                for ids in enc.encode_ordinary_batch(texts[i:i + 1024], num_threads=16):
                    chunk.append(bos)
                    chunk.extend(ids)
                np.asarray(chunk, dtype=np.uint16).tofile(f)
                n += len(chunk)
                if n >= MAX_TRAIN_TOKENS:
                    break
            if n >= MAX_TRAIN_TOKENS:
                break
    print(f"train tokens: {n} ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(enc, open(f"{out}/train/tokenizer.pkl", "wb"))

    vdocs = list(docs(val_path))
    half = len(vdocs) // 2
    for split, texts in (("validation", vdocs[:half]), ("holdout", vdocs[half:])):
        os.makedirs(f"{out}/{split}/private")
        os.makedirs(f"{out}/{split}/public")
        np.save(f"{out}/{split}/private/rows.npy", pack_rows(enc, bos, texts, EVAL_ROWS))
        np.save(f"{out}/{split}/private/token_bytes.npy", token_bytes)
    json.dump({"vocab_size": enc.n_vocab, "seq_len": SEQ_LEN, "train_tokens": n, "eval_rows": EVAL_ROWS,
               "train_shards": TRAIN_SHARDS, "val_shard": VAL_SHARD}, open(f"{out}/info.json", "w"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
