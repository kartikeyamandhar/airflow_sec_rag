"""Grounding verifier: does each cited span actually support its claim?

A single batched judge call scores every (claim, evidence) pair as SUPPORTED or
UNSUPPORTED. Verdicts are parsed conservatively: a claim counts as supported only if
the judge explicitly says so, so a missing or garbled verdict fails closed.
Faithfulness is the fraction of claims judged supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.grounding import Claim
from app.generation.llm import LLMClient

_VERDICT = re.compile(r"(\d+)\s*[:\-]\s*(SUPPORTED|UNSUPPORTED)", re.IGNORECASE)

_SYSTEM = (
    "You verify whether each claim is supported by its evidence. Use ONLY the "
    "evidence provided. A claim is SUPPORTED if the evidence states it or directly "
    "implies it; otherwise it is UNSUPPORTED. Respond with one line per claim in the "
    "form '<n>: SUPPORTED' or '<n>: UNSUPPORTED', and nothing else."
)


@dataclass(frozen=True)
class ClaimVerdict:
    index: int
    supported: bool


@dataclass(frozen=True)
class VerificationResult:
    verdicts: list[ClaimVerdict]
    faithfulness: float


class GroundingVerifier:
    """Judge whether each claim is entailed by its cited evidence."""

    def __init__(self, judge: LLMClient, max_tokens: int = 512) -> None:
        self._judge = judge
        self._max_tokens = max_tokens

    def verify(self, claims: list[Claim], evidences: list[str]) -> VerificationResult:
        if not claims:
            return VerificationResult(verdicts=[], faithfulness=0.0)
        user = self._build_prompt(claims, evidences)
        raw = self._judge.complete(system=_SYSTEM, user=user, max_tokens=self._max_tokens)
        supported = self._parse_supported(raw)
        verdicts = [
            ClaimVerdict(index=i + 1, supported=(i + 1) in supported) for i in range(len(claims))
        ]
        faithfulness = sum(v.supported for v in verdicts) / len(verdicts)
        return VerificationResult(verdicts=verdicts, faithfulness=faithfulness)

    def _build_prompt(self, claims: list[Claim], evidences: list[str]) -> str:
        blocks = [
            f"Claim {i}: {claim.text}\nEvidence {i}: {evidence}"
            for i, (claim, evidence) in enumerate(zip(claims, evidences, strict=True), start=1)
        ]
        return "\n\n".join(blocks) + "\n\nVerdicts:"

    def _parse_supported(self, raw: str) -> set[int]:
        return {
            int(match.group(1))
            for match in _VERDICT.finditer(raw)
            if match.group(2).upper() == "SUPPORTED"
        }
