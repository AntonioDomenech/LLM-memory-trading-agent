"""Owned system prompt for the frozen SEC-filing Gemma extractor.

Keeping the prompt in a dedicated source module gives the conceptual
``extractor_prompt`` role one exact repository owner.  The value is data only:
this module performs no I/O and grants no execution authority.
"""

from __future__ import annotations

from typing import Final


EXTRACTOR_SYSTEM_PROMPT: Final[str] = (
    "Extract only evidence-grounded relative business conditions from the anonymized "
    "current periodic filing and its optional anonymized prior same-form filing. Use "
    "only supplied C and P sentence identifiers. Return exactly the required JSON "
    "schema. Do not infer or name the issuer, date, security, price, return, forecast, "
    "benchmark, or trading action."
)


__all__ = ["EXTRACTOR_SYSTEM_PROMPT"]
