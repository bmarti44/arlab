"""Frozen reference arm (a): no adaptation. The base model answers each goal from the tool names alone."""
ARM = "adapter"


def adapt(transcript, tool_names, gen, train):
    return None
