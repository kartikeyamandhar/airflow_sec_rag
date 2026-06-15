"""Generation: grounded, span-cited answers (or an explicit refusal).

Builds a strict grounding prompt over retrieved chunks, calls the answer model,
enforces that every kept sentence cites a retrieved passage, refuses when the
context does not support an answer, and scores its own confidence.
"""
