# SEC RAG: Grounded Filings Answer Engine

A grounded answer engine over SEC filings (10-K, 10-Q, 8-K, earnings transcripts).
You ask a question over the filing corpus and get a numbers-correct, span-cited
answer with an "as of" date, or an explicit "not supported by any filing."

Every claim is cited to a source span. The system scores its own confidence,
refuses when nothing supports an answer, and blocks its own deploys when answer
accuracy regresses on a golden set.

> Status: Phase 4 complete (retrieval). The pipeline discovers, stores, parses, and
> embeds filings, then answers retrieval queries with hybrid (dense + BM25) search,
> metadata filtering, reranking, and comparison-question decomposition; grounded
> answer generation comes next.

## Two data planes

- Numbers come from XBRL via the EDGAR CompanyFacts API. They are structured and
  clean, and are never parsed out of HTML tables.
- Narrative (MD&A, risk factors) comes from the filing HTML. It is parsed and
  chunked. Hard parsing is confined to prose.

## Requirements

- Python 3.12 (provisioned automatically by uv; pinned in `.python-version`).
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- Docker, for the local Postgres index and for the test suite (testcontainers).
- git. An EDGAR identity and Cloudflare R2 credentials are needed for live
  acquisition; RunPod and an LLM key come in later phases (see `.env.example`).

## Quickstart

```bash
git clone https://github.com/kartikeyamandhar/airflow_sec_rag.git sec_rag
cd sec_rag

# Create the environment from the lockfile and install dev tooling.
uv sync

# Install the git pre-commit hooks (ruff, mypy, hygiene). One time.
uv run pre-commit install

# Run the full local gate (lint, types, tests). Must exit 0.
make check
```

`uv sync` followed by `make check` should take a fresh clone to a green gate.

## Configuration

Configuration is read from the environment or a local `.env` via
`pydantic-settings`. Copy the example and fill it in:

```bash
cp .env.example .env
```

Secrets are typed as `SecretStr`, so they are redacted in logs and `repr()`.
Do not commit `.env`. It is git-ignored; only `.env.example` (no values) is tracked.

For live acquisition set `EDGAR_IDENTITY` (your name and email; EDGAR has no key)
and the `R2_*` values, including `R2_ENDPOINT_URL`.

## Data acquisition (Phase 1)

Acquisition runs in two idempotent, resumable steps over a configured universe of
companies ([configs/universe.dev.yaml](configs/universe.dev.yaml)).

```bash
# 1. Start the local Postgres index (and Qdrant for the embedding step).
make db-up

# 2. Discover target filings (ticker -> CIK, enumerate 10-K/10-Q in the window)
#    and record them in the index.
uv run python -m scripts.discover_filings --config configs/universe.dev.yaml

# 3. Fetch and store raw artifacts: CompanyFacts JSON (numbers) per company and
#    the primary document (narrative) per filing, into object storage.
uv run python -m scripts.acquire_filings

# 4. Parse stored artifacts: CompanyFacts JSON into numeric facts, and each filing
#    into section-aware parent/child text chunks with citation offsets.
uv run python -m scripts.parse_filings

# 5. Embed the child chunks and load them into the Qdrant vector index.
#    Default backend is local CPU (fastembed); set EMBEDDING_BACKEND=runpod to use a
#    deployed RunPod GPU endpoint (see infra/runpod/).
uv run python -m scripts.index_chunks

# 6. Query the index: hybrid (dense + BM25) search with metadata filtering and
#    reranking. Returns cited chunks. (Grounded answer generation is a later phase.)
uv run python -m scripts.search "what are Apple's supply chain risks?" --ticker AAPL
```

All steps are safe to re-run: completed work is skipped via the index checkpoint.
Numbers come from the XBRL CompanyFacts API and narrative from the filing HTML.
Parsing keeps the planes separate: numeric facts (attributed to their filing) and
chunked text whose offsets map back to an exact, citable source span.

## Commands

All commands run through `uv` so everyone uses the same pinned toolchain. Use
`make` or `uv run` rather than bare `python`, `mypy`, or `ruff`.

| Command       | What it does                                       |
| ------------- | -------------------------------------------------- |
| `make setup`  | `uv sync` plus install pre-commit hooks            |
| `make format` | Auto-format and auto-fix (ruff). Mutating.         |
| `make lint`   | Lint and format-check (ruff). Non-mutating.        |
| `make type`   | Type-check (mypy strict).                          |
| `make test`   | Run tests with coverage (pytest).                  |
| `make check`  | `lint`, `type`, `test`. The full gate.             |
| `make db-up`  | Start the local Postgres index (docker compose).   |
| `make db-down`| Stop the local Postgres index.                     |

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the same `make`
targets on a clean runner, so local green and CI green stay in sync.

## Repository layout

```
app/        library code: clients, parsing, chunking, embedding, retrieval, generation, eval
configs/    pydantic-settings models, company lists, run configs
data/        local scratch only, never committed
notebooks/   experiments
scripts/     CLI entrypoints (each an idempotent, resumable job)
tests/       unit and integration
docs/        architecture, ADRs, per-phase specs, progress log
infra/       Dockerfiles, compose (Qdrant, Airflow), RunPod job specs
.github/     CI workflows
```

## License

MIT.
