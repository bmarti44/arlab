"""Frozen constants shared by PREPARE, the RUN harness, EVALUATE and the tests."""

# Pinned, cached, offline (no downloads anywhere in this pack).
MODEL_DIR = "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# ---- sizes (CALIBRATE IN THE GPU PILOT; a change here changes data_hash -> new data, new tag)
CONTEXT_TOKENS = 8192      # document prompt length target (chat head + log); every doc is within 128 tokens below it
N_ITEMS = 400              # items (one question per document) per split: validation and holdout

# Qwen3 chat format with thinking disabled (what apply_chat_template(enable_thinking=False) produces).
HEAD = "<|im_start|>user\n"                      # starts every prompt; part of the stored document tokens
# The assistant turn is prefilled with "Answer:" (a CPU check on 12 items: without it the base model often continues
# the log, "R-569 was moved to dock Hivail", or starts explaining; with it, it answers tersely).
ANSWER_PREFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
QUESTION = "Question: {q}" + ANSWER_PREFIX
IM_END, EOT = 151645, 151643                      # <|im_end|>, <|endoftext|>
STOP_IDS = (IM_END, EOT)
MAX_NEW = 12                                      # greedy answer tokens (answers are 1-6 tokens + <|im_end|>)
