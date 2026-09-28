"""Frozen reference arm no_ttt: the document in context, no test-time updates (plain long-context answering)."""
DOC_IN_CONTEXT = True


def adapt(model, ctx):
    return None
