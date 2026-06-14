"""Object storage for raw filing artifacts.

A small ``RawStore`` Protocol decouples business logic from the backend. The
local filesystem backend is used in tests and offline dev; the S3 backend targets
Cloudflare R2 in real runs. Storage keys are derived from validated identifiers
(see :mod:`app.storage.keys`), never from raw user input.
"""
