# SEC RAG: Grounded Filings Answer Engine

A grounded answer engine over SEC filings (10-K, 10-Q, 8-K, earnings transcripts).
You ask a question over the filing corpus and get a numbers-correct, span-cited
answer with an "as of" date, or an explicit "not supported by any filing."

Every claim is cited to a source span. The system scores its own confidence,
refuses when nothing supports an answer, and blocks its own deploys when answer
accuracy regresses on a golden set.

> Status: Phase 0 complete (scaffolding and environment). No application behavior
> yet.

## Two data planes

- Numbers come from XBRL via the EDGAR CompanyFacts API. They are structured and
  clean, and are never parsed out of HTML tables.
- Narrative (MD&A, risk factors) comes from the filing HTML. It is parsed and
  chunked. Hard parsing is confined to prose.

## Requirements

- Python 3.12 (provisioned automatically by uv; pinned in `.python-version`).
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- git. Docker, RunPod, R2, and an LLM key are needed in later phases (see
  `.env.example`).

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
