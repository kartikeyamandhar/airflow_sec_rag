"""Narrative parsing and chunking (the prose plane).

Extracts clean text from filing HTML, segments it into SEC item sections, and
chunks it into parent (section) and child (token-window) units with character
offsets that make later span citation exact.
"""
