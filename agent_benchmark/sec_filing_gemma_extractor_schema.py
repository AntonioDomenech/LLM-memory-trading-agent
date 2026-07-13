"""Owned JSON schema and enums for the SEC-filing Gemma extractor.

The schema is deliberately pure and deterministic.  It has no knowledge of a
stage, filing, issuer, market observation, model runtime, or filesystem path.
"""

from __future__ import annotations

import copy
from typing import Any, Final


EXTRACTOR_SCHEMA_VERSION: Final[str] = "sec-filing-extractor-v1"
DIMENSION_NAMES: Final[tuple[str, ...]] = (
    "demand",
    "pricing_power",
    "gross_margin",
    "operating_cost_pressure",
    "capital_allocation",
    "liquidity",
    "forward_guidance",
    "supply_chain",
    "legal_regulatory",
    "management_uncertainty",
)
FLAG_NAMES: Final[tuple[str, ...]] = (
    "new_material_risk",
    "guidance_withdrawn",
    "liquidity_stress",
    "restructuring_or_impairment",
    "internal_control_weakness",
    "management_transition",
)
ADVERSE_FLAG_NAMES: Final[tuple[str, ...]] = FLAG_NAMES[:-1]
CURRENT_IMPACTS: Final[frozenset[str]] = frozenset(
    {"favorable", "neutral", "unfavorable", "mixed", "not_stated"}
)
COMPARATIVE_CHANGES: Final[frozenset[str]] = frozenset(
    {
        "improving",
        "stable",
        "deteriorating",
        "mixed",
        "not_comparable",
        "not_stated",
    }
)
DOCUMENT_QUALITIES: Final[frozenset[str]] = frozenset(
    {"usable", "thin", "unusable"}
)


def build_extractor_json_schema() -> dict[str, Any]:
    """Return a detached copy of the exact bounded extractor output schema."""

    dimension_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["current_impact", "change_vs_prior", "evidence_sentence_ids"],
        "properties": {
            "current_impact": {"type": "string", "enum": sorted(CURRENT_IMPACTS)},
            "change_vs_prior": {
                "type": "string",
                "enum": sorted(COMPARATIVE_CHANGES),
            },
            "evidence_sentence_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[CP][0-9]{4}$"},
                "uniqueItems": True,
            },
        },
    }
    flag_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["present", "evidence_sentence_ids"],
        "properties": {
            "present": {"type": "boolean"},
            "evidence_sentence_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[CP][0-9]{4}$"},
                "uniqueItems": True,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "document_quality", "dimensions", "flags"],
        "properties": {
            "schema_version": {"const": EXTRACTOR_SCHEMA_VERSION},
            "document_quality": {
                "type": "string",
                "enum": sorted(DOCUMENT_QUALITIES),
            },
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "required": list(DIMENSION_NAMES),
                "properties": {
                    name: copy.deepcopy(dimension_schema) for name in DIMENSION_NAMES
                },
            },
            "flags": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FLAG_NAMES),
                "properties": {
                    name: copy.deepcopy(flag_schema) for name in FLAG_NAMES
                },
            },
        },
    }


__all__ = [
    "ADVERSE_FLAG_NAMES",
    "COMPARATIVE_CHANGES",
    "CURRENT_IMPACTS",
    "DIMENSION_NAMES",
    "DOCUMENT_QUALITIES",
    "EXTRACTOR_SCHEMA_VERSION",
    "FLAG_NAMES",
    "build_extractor_json_schema",
]
