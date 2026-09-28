"""Editable surface: turn one world's exploration transcript into a per-world LoRA ("sleep" consolidation).

BASELINE = naive next-token LoRA on the raw transcript text (the SEAL / CodeUpdateArena control, expected to close
little of the gap to the transcript-in-context arm).

The frozen harness calls, once per world (fresh model, adapters reset between worlds):
    adapt(transcript, tool_names, gen, train) -> adapter from train(...) | None
transcript: list of {"call": 'name(arg, ...)', "obs": 'output or "error <code>"'} in the order they happened;
tool_names: this world's tool names; gen / train: see frozen_run/engine.py and frozen_run/trainer.py.
adapt() only ever receives the world being adapted; the evaluator scores the adapter on that same world.
At evaluation the adapted model sees only the system prompt with the tool names and one goal (no transcript) and must
write a program; see frozen_run/common.py for the exact format (gen.task_prompt(goal) builds the user message).
"""

LORA = {"rank": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lr": 2e-4, "epochs": 8, "batch_size": 2, "max_len": 1024}


def adapt(transcript, tool_names, gen, train):
    return train([{"text": gen.transcript_text}], LORA)
