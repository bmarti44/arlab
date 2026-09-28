"""PREPARE for latent-arch (mounted only here, at /prepare; RUN cannot regenerate programs or labels).

Text: nanochat-lite's recipe unchanged (climbmix shards, same pins, vocab-8192 rustbpe tokenizer, uint16 stream;
validation/holdout text rows = the two halves of the pinned upstream validation shard).
Programs: progen.py, one generator seed family per split, hash-deduplicated across splits (eval first, then train).
Secret vocabulary relabelling: a random permutation of the 8192 token ids (os.urandom) is applied to ALL token data;
token_bytes are permuted to match. The permutation and the tokenizer are kept in audit/ (read by the pack tests only;
never mounted into RUN or EVALUATE).

Layout under --out:
  train/tokens.bin, train/programs.bin (uint16), train/program_starts.npy (int64)   the only data RUN sees
  {validation,holdout}/private/text_rows.npy (2048, 1025) uint16, token_bytes.npy (8192,) int32,
                              programs.npz (items: tokens, answer, k, group, floors, counterfactual links)
  audit/vocab_perm.npy, audit/tokenizer.pkl, audit/train_k.npy          tests only
  splits.json, info.json
"""
import argparse
import json
import os
import pickle
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken

import progen

# ---- nanochat-lite pins (identical recipe)
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
TRAIN_SHARDS = list(range(8))
VAL_SHARD = 6542
VOCAB_SIZE = 8192
SEQ_LEN = 1024
TOKENIZER_CHARS = 400_000_000
MAX_TRAIN_TOKENS = 400_000_000
EVAL_ROWS = 2048
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS = "<|reserved_0|>"
RAW = "/tmp/raw"

# ---- programs
N_TRAIN = 2_000_000
N_ID, N_DEPTH, N_EXT, N_CF = 2000, 2000, 500, 500      # per eval split
SEEDS = {"validation": 2001, "holdout": 3001, "train": 1001}
GROUPS = {"id": 0, "depth": 1, "ext": 2, "cf": 3}


def fetch(i):
    path = f"{RAW}/shard_{i:05d}.parquet"
    if os.path.exists(path):
        return path
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


def train_tokenizer(train_paths):
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
    return tiktoken.Encoding(name="rustbpe", pat_str=tok.get_pattern(), mergeable_ranks=ranks,
                             special_tokens={n: len(ranks) + i for i, n in enumerate(SPECIAL_TOKENS)})


def piece_tokens(enc) -> tuple[np.ndarray, dict]:
    """Original token id of every progen piece. Each piece must be exactly one token; the few numbers the BPE did
    not merge (nanochat-lite's tokenizer: only "79") get an unused reserved special token instead (recorded in
    info.json). Programs are written as token ids directly, never re-tokenized, so this is exact."""
    ids, fallback = [], {}
    spare = iter(SPECIAL_TOKENS[1:])
    for p in progen.PIECES:
        t = enc.encode_ordinary(p)
        if len(t) != 1:
            assert p.isdigit(), f"piece {p!r} is {len(t)} tokens: {t}"
            name = next(spare)  # StopIteration = more than 3 unmerged numbers: the tokenizer changed, fix the pack
            fallback[p] = name
            t = [enc.encode_single_token(name)]
        ids.append(t[0])
    assert len(set(ids)) == len(ids)
    return np.asarray(ids, dtype=np.int64), fallback


def program_rows(progs, piece_tok, bos):
    """(n, 1 + n_pieces) prompt token ids: BOS + statements + `q?` (original ids)."""
    idx = np.array([[progen.PIECE_ID[x] for x in progen.pieces(p)] for p in progs], dtype=np.int64)
    return np.concatenate([np.full((len(progs), 1), bos), piece_tok[idx]], axis=1)


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
    enc = train_tokenizer(train_paths)
    bos = enc.encode_single_token(BOS)
    token_bytes = np.array([0 if enc.decode([i]) in SPECIAL_TOKENS else len(enc.decode_single_token_bytes(i))
                            for i in range(enc.n_vocab)], dtype=np.int32)
    assert enc.n_vocab == VOCAB_SIZE
    piece_tok, fallback = piece_tokens(enc)
    print(f"tokenizer trained in {time.time() - t0:.0f}s (vocab {enc.n_vocab})", flush=True)

    # ---- the secret relabelling: stored id = perm[original id]
    perm = np.random.default_rng(int.from_bytes(os.urandom(16), "little")).permutation(VOCAB_SIZE).astype(np.int64)
    tb_perm = np.empty_like(token_bytes)
    tb_perm[perm] = token_bytes
    os.makedirs(f"{out}/audit")
    np.save(f"{out}/audit/vocab_perm.npy", perm)
    pickle.dump(enc, open(f"{out}/audit/tokenizer.pkl", "wb"))

    # ---- text stream (relabelled)
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
                perm[np.asarray(chunk, dtype=np.int64)].astype(np.uint16).tofile(f)
                n += len(chunk)
                if n >= MAX_TRAIN_TOKENS:
                    break
            if n >= MAX_TRAIN_TOKENS:
                break
    print(f"train text tokens: {n} ({time.time() - t0:.0f}s)", flush=True)

    # ---- eval programs first (validation, holdout), then train excluding every eval hash
    seen: set[int] = set()
    evals = {}
    for split in ("validation", "holdout"):
        s = SEEDS[split]
        g = {"id": progen.generate(s * 10 + 1, N_ID, progen.ID_K, balanced=True, seen=seen),
             "depth": progen.generate(s * 10 + 2, N_DEPTH, progen.DEPTH_K, balanced=True, seen=seen),
             "ext": progen.generate(s * 10 + 3, N_EXT, progen.EXT_K, balanced=True, seen=seen)}
        main = g["id"] + g["depth"]
        rng = random.Random(s * 10 + 4)
        g["cf"], cf_of = [], []
        for j in range(0, len(main), len(main) // N_CF):  # every 8th main item gets a counterfactual twin
            while True:
                c = progen.counterfactual(main[j], rng)
                h = progen.norm_hash(c)
                if h not in seen:
                    break
            seen.add(h)
            g["cf"].append({**c, "hash": h})
            cf_of.append(j)
        evals[split] = (g, cf_of)
    eval_hashes = set(seen)
    rng = random.Random(SEEDS["train"])
    train, train_k, dup_eval, dup_train = [], [], 0, 0  # train programs as bytes of piece ids (prompt + answer)
    while len(train) < N_TRAIN:
        p = progen.make_program(rng, rng.randint(*progen.TRAIN_K))
        h = progen.norm_hash(p)
        if h in seen:
            dup_eval += h in eval_hashes
            dup_train += h not in eval_hashes
            continue
        seen.add(h)
        a = progen.annotate(p)
        train.append(bytes([progen.PIECE_ID[x] for x in progen.pieces(p)] + [progen.PIECE_ID[str(a["answer"])]]))
        train_k.append(p["k"])
    print(f"programs generated ({time.time() - t0:.0f}s); train redraws: {dup_eval} eval collisions, "
          f"{dup_train} train duplicates", flush=True)

    # ---- train program stream: BOS + prompt + answer per program, relabelled
    idx = np.frombuffer(b"".join(train), dtype=np.uint8).reshape(len(train), -1)
    full = perm[np.concatenate([np.full((len(train), 1), bos), piece_tok[idx]], axis=1)].astype(np.uint16)
    full.reshape(-1).tofile(f"{out}/train/programs.bin")
    np.save(f"{out}/train/program_starts.npy", np.arange(len(train), dtype=np.int64) * full.shape[1])
    np.save(f"{out}/audit/train_k.npy", np.array(train_k, dtype=np.uint8))
    del idx, full

    # ---- eval splits: text rows + permuted token_bytes + programs, all private
    vdocs = list(docs(val_path))
    half = len(vdocs) // 2
    floors = {}
    for split, texts in (("validation", vdocs[:half]), ("holdout", vdocs[half:])):
        os.makedirs(f"{out}/{split}/private")
        np.save(f"{out}/{split}/private/text_rows.npy", perm[pack_rows(enc, bos, texts, EVAL_ROWS)].astype(np.uint16))
        np.save(f"{out}/{split}/private/token_bytes.npy", tb_perm)
        g, cf_of = evals[split]
        items, group, link = [], [], []
        for name in ("id", "depth", "ext", "cf"):
            items += g[name]
            group += [GROUPS[name]] * len(g[name])
        link = [-1] * (len(items) - len(cf_of)) + cf_of  # cf item -> index of its original (main items come first)
        pid = [progen.PIECE_ID[str(p["answer"])] for p in items]
        np.savez(f"{out}/{split}/private/programs.npz",
                 tokens=perm[program_rows(items, piece_tok, bos)].astype(np.int64),
                 answer=perm[piece_tok[pid]].astype(np.int64),
                 value=np.array([p["answer"] for p in items]), k=np.array([p["k"] for p in items]),
                 group=np.array(group), cf_of=np.array(link),
                 h_last_const=np.array([p["h_last_const"] for p in items]),
                 h_own_const=np.array([p["h_own_const"] for p in items]),
                 h_root=np.array([p["h_root"] for p in items]),
                 hash=np.array([p["hash"] for p in items], dtype=np.uint64),
                 text=np.array([progen.to_text(p) for p in items]))
        main = g["id"] + g["depth"]
        floors[split] = {h: sum(p[h] == p["answer"] for p in main) / len(main)
                         for h in ("h_last_const", "h_own_const", "h_root")}
        floors[split]["modal"] = Counter(p["answer"] for p in main).most_common(1)[0][1] / len(main)

    json.dump({"validation": N_ID + N_DEPTH, "holdout": N_ID + N_DEPTH}, open(f"{out}/splits.json", "w"))
    info = {"vocab_size": enc.n_vocab, "seq_len": SEQ_LEN, "train_text_tokens": n, "eval_rows": EVAL_ROWS,
            "train_shards": TRAIN_SHARDS, "val_shard": VAL_SHARD, "difficulty": progen.DIFFICULTY,
            "train_k": progen.TRAIN_K, "id_k": progen.ID_K, "depth_k": progen.DEPTH_K, "ext_k": progen.EXT_K,
            "n_train_programs": len(train), "piece_fallback": fallback, "program_tokens": int(1 + progen.n_pieces() + 1),
            "n_eval": {"id": N_ID, "depth": N_DEPTH, "ext": N_EXT, "cf": N_CF}, "seeds": SEEDS,
            "train_redraws": {"eval_collisions": dup_eval, "train_duplicates": dup_train}, "floors": floors,
            "example": progen.to_text(evals["validation"][0]["id"][0])}
    json.dump(info, open(f"{out}/info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
