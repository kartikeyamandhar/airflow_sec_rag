"""Configuration surface for SEC RAG.

Holds the typed settings model (``configs.settings``) plus, in later phases,
company lists and run configs. Settings are loaded from the environment / ``.env``
via pydantic-settings; no secrets are ever hard-coded here.
"""
