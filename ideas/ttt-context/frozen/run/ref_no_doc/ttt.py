"""Frozen reference arm no_doc (pilot floor): the question alone, no document, no updates. Not a pack.yaml reference
(see IDEA.md: as a reference its paired item sd would dominate the runner's power check); run it with build/pilot.sh."""


def adapt(model, ctx):
    return {"weights": {}, "doc_in_context": False}
