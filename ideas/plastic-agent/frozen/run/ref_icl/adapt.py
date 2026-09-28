"""Frozen reference arm (b): no adaptation; the evaluator puts the whole exploration transcript in the prompt
(in-context learning). It fails the prefill_tokens guard by design: it is the arm the surface should match at a
fraction of the inference tokens, not a candidate."""
ARM = "icl"


def adapt(transcript, tool_names, gen, train):
    return None
