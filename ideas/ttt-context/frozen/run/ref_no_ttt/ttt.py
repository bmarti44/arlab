"""Frozen reference arm no_ttt: the document in context, no test-time updates (plain long-context answering)."""


def adapt(model, ctx):
    return {"weights": {}, "doc_in_context": True}
