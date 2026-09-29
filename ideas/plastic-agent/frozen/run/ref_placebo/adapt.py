"""Frozen reference arm (e), placebo: exactly the baseline surface's adapt() (naive next-token LoRA on the transcript).
The harness recognises this file by its sha256 and hands world i the transcript of its TWIN (the same tool names and
argument vocabulary, other semantics); the evaluator scores the resulting adapter on world i, like every arm. It
measures what a LoRA of the right names and format but the wrong facts does to success and to the forgetting battery.
"""

LORA = {"rank": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lr": 2e-4, "epochs": 8, "batch_size": 2, "max_len": 1024}


def adapt(transcript, tool_names, gen, train):
    return train([{"text": gen.transcript_text}], LORA)
