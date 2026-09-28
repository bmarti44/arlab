"""Frozen reference arm no_doc (pilot floor): the question alone, no document, no updates. Not a pack.yaml reference
(see IDEA.md: as a reference its paired item sd would dominate the runner's power check); run it with build/pilot.sh."""
DOC_IN_CONTEXT = False
PREFILL_DOC = False


def adapt(model, ctx):
    return None
