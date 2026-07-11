from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_benchmark.sec_audit_evaluation import (
    EXPECTED_SLOT_IDS,
    FilingAuditResult,
    NormalizedFilingIdentity,
    NormalizedIndexIdentity,
    PeriodicCoverageResult,
    REQUIRED_PERIODIC_YEARS,
    evaluate_sec_audit,
)
from agent_benchmark.sec_audit_selection import HISTORICAL_ROLE, STRUCTURAL_ROLE


def _accession(year: int, sequence: int, *, submitter: int = 320193) -> str:
    return f"{submitter:010d}-{year % 100:02d}-{sequence:06d}"


def _identity(slot_id: str, position: int) -> NormalizedFilingIdentity:
    if slot_id.startswith("core-"):
        _, year_text, category = slot_id.split("-", 2)
        year = int(year_text)
        form = {
            "periodic": "10-Q",
            "8-k": "8-K",
            "def-14a": "DEF 14A",
        }[category]
    else:
        edge_values = {
            "edge-earliest-2000": (2000, "S-8"),
            "edge-first-xbrl": (2008, "S-3"),
            "edge-first-amendment": (2006, "10-K/A"),
            "edge-first-after-1730-et": (2007, "S-4"),
            "edge-first-differing-submitter-cik": (2003, "SC 13G"),
            "edge-first-anomaly": (2010, "8-A12B"),
        }
        year, form = edge_values[slot_id]
    return NormalizedFilingIdentity(
        accession=_accession(year, position + 1),
        form=form,
        filing_date=f"{year:04d}-02-01",
    )


def _passing_inputs() -> tuple[list[FilingAuditResult], list[PeriodicCoverageResult]]:
    slots: list[FilingAuditResult] = []
    for position, slot_id in enumerate(EXPECTED_SLOT_IDS):
        identity = _identity(slot_id, position)
        year = int(identity.filing_date[:4])
        role = STRUCTURAL_ROLE if year >= 2019 else HISTORICAL_ROLE
        slots.append(
            FilingAuditResult(
                slot_id=slot_id,
                status="selected",
                evidence_role=role,
                selected=identity,
                submissions=identity,
                master_idx=identity,
                sgml_header=identity,
                index_metadata=NormalizedIndexIdentity(
                    identity.accession, identity.subject_cik
                ),
                # The frozen gate allows exactly one failure in each measure.
                exact_acceptance_timestamp=position != 0,
                usable_visible_text=position != 1,
            )
        )
    periodic = [
        PeriodicCoverageResult(
            year=year,
            accessions=(_accession(year, 900000 + position),),
        )
        for position, year in enumerate(REQUIRED_PERIODIC_YEARS)
    ]
    return slots, periodic


def test_all_frozen_gates_pass_with_23_of_24_thresholds() -> None:
    slots, periodic = _passing_inputs()
    report = evaluate_sec_audit(slots, periodic)
    assert report["overall_pass"] is True
    assert report["required_sample_count"] == 24
    assert report["evidence_classification"] == (
        "sec_source_audit_only_not_model_or_trading_evidence"
    )
    gates = report["gates"]
    selection = gates["exact_24_unique_selected_filings_and_roles"]
    assert selection["passed"] is True
    assert selection["selected_count"] == 24
    assert selection["unique_selected_accession_count"] == 24
    assert selection["role_counts"] == {
        HISTORICAL_ROLE: 18,
        STRUCTURAL_ROLE: 6,
    }
    assert gates["all_24_cross_source_identities_agree"]["agreement_count"] == 24
    assert gates["at_least_23_exact_acceptance_timestamps"]["observed_count"] == 23
    assert gates["at_least_23_usable_visible_texts"]["observed_count"] == 23
    assert gates["periodic_coverage_2000_2024"]["covered_years"] == list(
        range(2000, 2025)
    )
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_report_is_deterministic_for_reversed_normalized_inputs() -> None:
    slots, periodic = _passing_inputs()
    first = evaluate_sec_audit(slots, periodic)
    second = evaluate_sec_audit(reversed(slots), reversed(periodic))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cross_source_mismatch_fails_with_per_slot_evidence() -> None:
    slots, periodic = _passing_inputs()
    target = slots[4]
    wrong = replace(target.selected, form="10-Q")
    slots[4] = replace(target, master_idx=wrong)
    report = evaluate_sec_audit(slots, periodic)
    gate = report["gates"]["all_24_cross_source_identities_agree"]
    assert report["overall_pass"] is False
    assert gate["agreement_count"] == 23
    evidence = {item["slot_id"]: item for item in gate["filings"]}
    assert evidence[target.slot_id]["mismatched_sources"] == ["master_idx"]
    assert evidence[target.slot_id]["missing_sources"] == []


def test_missing_source_identity_fails_agreement() -> None:
    slots, periodic = _passing_inputs()
    slots[3] = replace(slots[3], sgml_header=None)
    gate = evaluate_sec_audit(slots, periodic)["gates"][
        "all_24_cross_source_identities_agree"
    ]
    evidence = {item["slot_id"]: item for item in gate["filings"]}
    assert gate["passed"] is False
    assert evidence[slots[3].slot_id]["missing_sources"] == ["sgml_header"]


def test_two_inexact_timestamps_fail_the_23_of_24_gate() -> None:
    slots, periodic = _passing_inputs()
    slots[2] = replace(slots[2], exact_acceptance_timestamp=False)
    report = evaluate_sec_audit(slots, periodic)
    gate = report["gates"]["at_least_23_exact_acceptance_timestamps"]
    assert gate["observed_count"] == 22
    assert gate["passed"] is False
    assert report["overall_pass"] is False


def test_two_unusable_texts_fail_the_23_of_24_gate() -> None:
    slots, periodic = _passing_inputs()
    slots[2] = replace(slots[2], usable_visible_text=False)
    report = evaluate_sec_audit(slots, periodic)
    gate = report["gates"]["at_least_23_usable_visible_texts"]
    assert gate["observed_count"] == 22
    assert gate["passed"] is False


def test_explicit_gap_cannot_be_hidden_by_an_extra_substitute() -> None:
    slots, periodic = _passing_inputs()
    missing = slots[5]
    slots[5] = FilingAuditResult(
        slot_id=missing.slot_id,
        status="gap",
        evidence_role=missing.evidence_role,
        selected=None,
        submissions=None,
        master_idx=None,
        sgml_header=None,
        index_metadata=None,
        exact_acceptance_timestamp=False,
        usable_visible_text=False,
        missing_reason="no_filing_matches_exact_slot",
    )
    substitute_identity = NormalizedFilingIdentity(
        accession=_accession(2017, 999999),
        form="8-K",
        filing_date="2017-03-01",
    )
    slots.append(
        FilingAuditResult(
            slot_id="extra-substitute-slot",
            status="selected",
            evidence_role=HISTORICAL_ROLE,
            selected=substitute_identity,
            submissions=substitute_identity,
            master_idx=substitute_identity,
            sgml_header=substitute_identity,
        index_metadata=NormalizedIndexIdentity(
            substitute_identity.accession, substitute_identity.subject_cik
        ),
            exact_acceptance_timestamp=True,
            usable_visible_text=True,
        )
    )
    report = evaluate_sec_audit(slots, periodic)
    selection = report["gates"]["exact_24_unique_selected_filings_and_roles"]
    missing_gate = report["gates"][
        "explicit_missing_slots_fail_without_substitution"
    ]
    assert report["overall_pass"] is False
    assert missing.slot_id in selection["gap_slot_ids"]
    assert selection["extra_slot_ids"] == ["extra-substitute-slot"]
    assert missing_gate["passed"] is False
    assert missing_gate["substitution_allowed"] is False
    assert missing_gate["missing_evidence"] == [
        {
            "slot_id": missing.slot_id,
            "status": "gap",
            "reason": "no_filing_matches_exact_slot",
        }
    ]


def test_duplicate_accession_fails_unique_selection_gate() -> None:
    slots, periodic = _passing_inputs()
    first = slots[0].selected
    assert first is not None
    slots[1] = replace(
        slots[1],
        selected=first,
        submissions=first,
        master_idx=first,
        sgml_header=first,
        index_metadata=NormalizedIndexIdentity(first.accession, first.subject_cik),
    )
    gate = evaluate_sec_audit(slots, periodic)["gates"][
        "exact_24_unique_selected_filings_and_roles"
    ]
    assert gate["passed"] is False
    assert gate["unique_selected_accession_count"] == 23
    assert gate["duplicate_accessions"] == [first.accession]


def test_role_must_match_pre2019_or_structural_only_policy() -> None:
    slots, periodic = _passing_inputs()
    slots[0] = replace(slots[0], evidence_role=STRUCTURAL_ROLE)
    gate = evaluate_sec_audit(slots, periodic)["gates"][
        "exact_24_unique_selected_filings_and_roles"
    ]
    assert gate["passed"] is False
    assert gate["invalid_role_slot_ids"] == [slots[0].slot_id]


def test_core_slot_year_and_form_meaning_cannot_be_faked() -> None:
    slots, periodic = _passing_inputs()
    target = slots[1]
    wrong = replace(target.selected, form="10-Q")
    slots[1] = replace(
        target,
        selected=wrong,
        submissions=wrong,
        master_idx=wrong,
        sgml_header=wrong,
        index_metadata=NormalizedIndexIdentity(wrong.accession, wrong.subject_cik),
    )
    report = evaluate_sec_audit(slots, periodic)
    gate = report["gates"]["core_year_form_and_identity_slot_semantics"]
    assert gate["passed"] is False
    assert gate["failures"][0]["slot_id"] == target.slot_id
    assert report["overall_pass"] is False


def test_periodic_coverage_requires_each_bounded_year_once_and_nonempty() -> None:
    slots, periodic = _passing_inputs()
    periodic = [item for item in periodic if item.year != 2007]
    periodic = [
        replace(item, accessions=()) if item.year == 2012 else item
        for item in periodic
    ]
    periodic.append(
        PeriodicCoverageResult(
            year=2025,
            accessions=(_accession(2025, 999998),),
        )
    )
    gate = evaluate_sec_audit(slots, periodic)["gates"][
        "periodic_coverage_2000_2024"
    ]
    assert gate["passed"] is False
    assert gate["missing_year_records"] == [2007]
    assert gate["years_without_periodic_filing"] == [2012]
    assert gate["unexpected_year_records"] == [2025]


def test_missing_expected_slot_and_duplicate_slot_both_fail() -> None:
    slots, periodic = _passing_inputs()
    removed = slots.pop(7)
    slots.append(replace(slots[0]))
    gate = evaluate_sec_audit(slots, periodic)["gates"][
        "exact_24_unique_selected_filings_and_roles"
    ]
    assert gate["passed"] is False
    assert gate["missing_slot_ids"] == [removed.slot_id]
    assert gate["duplicate_slot_ids"] == [slots[0].slot_id]


def test_inputs_must_already_be_normalized() -> None:
    with pytest.raises(ValueError, match="boolean"):
        FilingAuditResult(
            slot_id=EXPECTED_SLOT_IDS[0],
            status="selected",
            evidence_role=HISTORICAL_ROLE,
            selected=None,
            submissions=None,
            master_idx=None,
            sgml_header=None,
            index_metadata=None,
            exact_acceptance_timestamp=1,  # type: ignore[arg-type]
            usable_visible_text=True,
        )
    with pytest.raises(ValueError, match="Apple audit"):
        NormalizedFilingIdentity(
            accession=_accession(2000, 1),
            form="10-K",
            filing_date="2000-01-01",
            subject_cik="0000789019",
        )
