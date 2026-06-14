"""EDGAR access: the only layer that talks to the SEC over the network.

The ``EdgarClient`` Protocol is the seam the rest of the system depends on, so
discovery and acquisition can be tested with an in-memory fake. The concrete
``EdgartoolsClient`` uses edgartools for discovery plumbing (ticker to CIK,
filing enumeration with pagination) and a polite, rate-limited httpx client for
raw byte retrieval.
"""
