from __future__ import annotations

from collections.abc import Iterator, Mapping
import copy
import hashlib
import json
import string

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    EXTRACTOR_REQUEST_VERSION,
    MANDATORY_IDENTITY_TERMS,
    MAX_INPUT_BYTES,
    MAX_SENTENCE_CHARACTERS,
    MAX_SENTENCES,
    PREPROCESSOR_VERSION,
    build_extractor_model_payload,
    build_redacted_input_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    validate_extractor_request,
)
from agent_benchmark.sec_filing_gemma_preprocessor import (
    CANONICAL_EXECUTIVE_IDENTITY_TERMS,
    CANONICAL_IDENTITY_LEXICON,
    CANONICAL_IDENTITY_LEXICON_SHA256,
    MAX_CURRENT_SENTENCES,
    MAX_OWNED_NORMALIZED_SOURCE_BYTES,
    MAX_PRIOR_SENTENCES,
    OWNED_PREPROCESSING_RECEIPT_SCHEMA_VERSION,
    PREPROCESSED_EVENT_SCHEMA_VERSION,
    SecFilingGemmaPreprocessorError,
    build_owned_preprocessing_receipt,
    preprocess_filing_event,
    validate_owned_preprocessing_receipt,
    validate_preprocessed_event,
)


IDENTITY_LEXICON = CANONICAL_IDENTITY_LEXICON


class _SecondReadFlippingEvent(Mapping[str, object]):
    """Expose one canonical read, then forge one top-level field."""

    def __init__(self, value: dict, *, field: str, forged: object) -> None:
        self._value = value
        self._field = field
        self._forged = forged
        self.read_counts: dict[str, int] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __getitem__(self, key: str) -> object:
        count = self.read_counts.get(key, 0) + 1
        self.read_counts[key] = count
        if key == self._field and count > 1:
            return self._forged
        return self._value[key]


def _texts(artifact: dict, prefix: str | None = None) -> list[str]:
    return [
        sentence["text"]
        for sentence in artifact["sentences"]
        if prefix is None or sentence["id"].startswith(prefix)
    ]


def _alpha_suffix(number: int) -> str:
    # Deterministic alphabetic labels keep candidate sentences distinct without
    # introducing absolute numbers that the preprocessor must redact.
    first, second = divmod(number, len(string.ascii_lowercase))
    return string.ascii_lowercase[first] + string.ascii_lowercase[second]


def _normalized_source(relative_path: str, text: str) -> dict:
    payload = text.encode("utf-8")
    return {
        "relative_path": relative_path,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _owned_preprocessing_case() -> tuple[dict, dict]:
    current = "Demand improved while liquidity remained stable."
    prior = "Demand had weakened while supply risk increased."
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    kwargs = {
        "scope_kind": "development_root",
        "scope_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "model_execution_claim_sha256": "3" * 64,
        "sec_reader_receipt_sha256": "4" * 64,
        "carry_in_reader_receipt_sha256": None,
        "stage": "development",
        # Event order is chronological universe order, while file ordinals are
        # acquisition-plan order.  They are intentionally independent.
        "event_ordinal": 7,
        "accession_number": "0000320193-00-000002",
        "form": "10-Q",
        "current_normalized_source": _normalized_source(
            "document-0002.normalized.txt",
            current,
        ),
        "prior_same_form_normalized_source": _normalized_source(
            "document-0009.normalized.txt",
            prior,
        ),
        "prior_provenance_kind": "same_scope_document",
        "preprocessor_source_sha256": "5" * 64,
        "preprocessed_event": artifact,
        "current_normalized_text": current,
        "prior_same_form_normalized_text": prior,
    }
    return artifact, kwargs


def test_canonical_production_identity_lexicon_is_frozen_and_replayable() -> None:
    assert isinstance(CANONICAL_IDENTITY_LEXICON, tuple)
    assert CANONICAL_IDENTITY_LEXICON == tuple(sorted(CANONICAL_IDENTITY_LEXICON))
    assert len(CANONICAL_IDENTITY_LEXICON) == len(set(CANONICAL_IDENTITY_LEXICON))
    assert set(MANDATORY_IDENTITY_TERMS).issubset(CANONICAL_IDENTITY_LEXICON)
    assert {"luca maestri", "steve jobs"}.issubset(CANONICAL_IDENTITY_LEXICON)
    assert CANONICAL_IDENTITY_LEXICON_SHA256 == canonical_sha256(
        list(CANONICAL_IDENTITY_LEXICON)
    )

    required_executive_terms = {
        "jobs",
        "steve jobs",
        "cook",
        "tim cook",
        "oppenheimer",
        "peter oppenheimer",
        "maestri",
        "luca maestri",
        "parekh",
        "kevan parekh",
        "williams",
        "jeff williams",
        "ive",
        "jony ive",
        "schiller",
        "phil schiller",
        "cue",
        "eddy cue",
        "federighi",
        "craig federighi",
        "srouji",
        "johny srouji",
        "o'brien",
        "deirdre o'brien",
        "khan",
        "sabih khan",
        "ternus",
        "john ternus",
        "joswiak",
        "greg joswiak",
        "forstall",
        "scott forstall",
        "mansfield",
        "bob mansfield",
        "ahrendts",
        "angela ahrendts",
        "riccio",
        "dan riccio",
        "levinson",
        "arthur levinson",
    }
    assert required_executive_terms.issubset(CANONICAL_EXECUTIVE_IDENTITY_TERMS)
    assert set(CANONICAL_EXECUTIVE_IDENTITY_TERMS).issubset(
        CANONICAL_IDENTITY_LEXICON
    )


def test_executive_variants_and_unknown_honorific_people_are_redacted() -> None:
    current = (
        "Cook said customer demand improved. "
        "Maestri described liquidity risk. "
        "Jeff Williams said supply risk increased. "
        "Kevan Parekh said operating costs declined. "
        "Dr. Rowan Quill described regulatory uncertainty. "
        "Ms. Zephyr said inventory risk remained stable."
    )

    first = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    output = " ".join(_texts(first))
    lowered = output.casefold()

    assert replay == first
    assert output.count("[IDENTITY]") >= 6
    for leaked in (
        "cook",
        "maestri",
        "jeff williams",
        "kevan parekh",
        "rowan",
        "quill",
        "zephyr",
    ):
        assert leaked not in lowered
    assert "customer demand improved" in lowered
    assert "operating costs declined" in lowered
    assert first["redaction_report"]["identity_matches_remaining"] == 0


def test_honorifics_consume_the_complete_shared_bounded_person_name() -> None:
    current = (
        "Mr. Anna van der Meer resigned while demand improved. "
        "Mr. Susan Q. Wagner resigned while liquidity remained stable. "
        "Dr. John Ronald Reuel resigned while supply risk increased."
    )

    first = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    output = " ".join(_texts(first))
    lowered = output.casefold()

    assert replay == first
    assert output.count("[IDENTITY] resigned") == 3
    assert "[IDENTITY] resigned while demand improved" in output
    assert "[IDENTITY] resigned while liquidity remained stable" in output
    assert "[IDENTITY] resigned while supply risk increased" in output
    for leaked in (
        "anna",
        "meer",
        "susan",
        "wagner",
        "john",
        "ronald",
        "reuel",
    ):
        assert leaked not in lowered
    assert first["redaction_report"]["identity_matches_remaining"] == 0


def test_unlisted_people_are_redacted_by_title_and_speech_context() -> None:
    current = (
        "Susan Wagner said demand improved. "
        "Monica Lozano said liquidity remained stable. "
        "Chief Executive Officer Susan Wagner said demand improved. "
        "Independent Director Avery North noted supply risk increased. "
        "CFO Jordan Lee explained operating costs declined. "
        "Chief Executive Officer Susan Wagner oversees operations and liquidity. "
        "Susan Wagner, Chief Executive Officer, noted customer demand improved. "
        "Susan Wagner was appointed Chief Executive Officer. "
        "CEO Susan de Wagner said inventory risk remained stable."
    )

    first = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    output = " ".join(_texts(first))
    lowered = output.casefold()

    assert replay == first
    assert output.count("[IDENTITY]") >= 9
    for leaked in (
        "susan",
        "wagner",
        "monica",
        "lozano",
        "avery",
        "north",
        "jordan",
        "lee",
    ):
        assert leaked not in lowered
    assert "[IDENTITY] said demand improved" in output
    assert "Chief Executive Officer [IDENTITY] said demand improved" in output
    assert "Independent Director [IDENTITY] noted supply risk increased" in output
    assert "CFO [IDENTITY] explained operating costs declined" in output
    assert (
        "Chief Executive Officer [IDENTITY] oversees operations and liquidity"
        in output
    )
    assert (
        "[IDENTITY], Chief Executive Officer, noted customer demand improved"
        in output
    )
    assert "[IDENTITY] was appointed Chief Executive Officer" in output
    assert "CEO [IDENTITY] said inventory risk remained stable" in output
    assert first["redaction_report"]["identity_matches_remaining"] == 0


def test_bounded_parenthetical_particle_and_transition_people_are_redacted() -> None:
    current = (
        "Susan Wagner (Chief Executive Officer) said demand improved. "
        "Anna van der Meer indicated liquidity remained stable. "
        "Anna van der Meer will serve as Chief Executive Officer. "
        "Monica Lozano joined as Director. "
        "Avery North named Chief Financial Officer. "
        "Jordan Lee elected Chair."
    )

    first = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    output = " ".join(_texts(first))
    lowered = output.casefold()

    assert replay == first
    assert output.count("[IDENTITY]") >= 6
    for leaked in (
        "susan",
        "wagner",
        "anna",
        "meer",
        "monica",
        "lozano",
        "avery",
        "north",
        "jordan",
        "lee",
    ):
        assert leaked not in lowered
    assert "[IDENTITY] (Chief Executive Officer) said demand improved" in output
    assert "[IDENTITY] indicated liquidity remained stable" in output
    assert "[IDENTITY] will serve as Chief Executive Officer" in output
    assert "[IDENTITY] joined as Director" in output
    assert "[IDENTITY] named Chief Financial Officer" in output
    assert "[IDENTITY] elected Chair" in output
    assert first["redaction_report"]["identity_matches_remaining"] == 0


def test_owned_preprocessing_receipt_is_deterministic_and_exactly_replayable() -> None:
    artifact, kwargs = _owned_preprocessing_case()

    first = build_owned_preprocessing_receipt(**kwargs)
    replay = build_owned_preprocessing_receipt(**copy.deepcopy(kwargs))

    assert replay == first
    assert first["schema_version"] == OWNED_PREPROCESSING_RECEIPT_SCHEMA_VERSION
    assert first["receipt_kind"] == "owned_filing_event_preprocessing"
    assert first["scope_kind"] == "development_root"
    assert first["scope_sha256"] == "1" * 64
    assert first["candidate_sha256"] == "2" * 64
    assert first["model_execution_claim_sha256"] == "3" * 64
    assert first["sec_reader_receipt_sha256"] == "4" * 64
    assert first["carry_in_reader_receipt_sha256"] is None
    assert first["event_ordinal"] == 7
    assert first["current_normalized_source"]["relative_path"] == (
        "document-0002.normalized.txt"
    )
    assert first["prior_same_form_normalized_source"]["relative_path"] == (
        "document-0009.normalized.txt"
    )
    assert first["canonical_identity_lexicon_sha256"] == (
        CANONICAL_IDENTITY_LEXICON_SHA256
    )
    assert first["preprocessed_event_sha256"] == artifact[
        "preprocessed_event_sha256"
    ]
    assert first["model_payload_sha256"] == artifact["model_payload_sha256"]
    assert "model_payload" not in first
    assert "redacted_input_manifest_sha256" not in first
    body = {key: value for key, value in first.items() if key != "receipt_sha256"}
    assert first["receipt_sha256"] == canonical_sha256(body)
    assert validate_owned_preprocessing_receipt(
        first,
        **kwargs,
    ) == first["receipt_sha256"]


def test_owned_preprocessing_receipt_detaches_flipping_event_mapping_once() -> None:
    artifact, kwargs = _owned_preprocessing_case()
    forged_hash = "f" * 64
    flipping = _SecondReadFlippingEvent(
        artifact,
        field="model_payload_sha256",
        forged=forged_hash,
    )
    kwargs["preprocessed_event"] = flipping

    receipt = build_owned_preprocessing_receipt(**kwargs)

    assert flipping.read_counts["model_payload_sha256"] == 1
    assert receipt["model_payload_sha256"] == artifact["model_payload_sha256"]
    assert receipt["model_payload_sha256"] != forged_hash


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("model_execution_claim_sha256", "a" * 64),
        ("sec_reader_receipt_sha256", "b" * 64),
        ("candidate_sha256", "c" * 64),
        ("event_ordinal", 3),
        ("accession_number", "0000320193-00-000003"),
        ("preprocessor_source_sha256", "d" * 64),
    ],
)
def test_owned_preprocessing_receipt_rejects_rehashed_mutation(
    field: str,
    replacement: object,
) -> None:
    _, kwargs = _owned_preprocessing_case()
    receipt = build_owned_preprocessing_receipt(**kwargs)
    changed = copy.deepcopy(receipt)
    changed[field] = replacement
    changed_body = {
        key: value for key, value in changed.items() if key != "receipt_sha256"
    }
    changed["receipt_sha256"] = canonical_sha256(changed_body)

    with pytest.raises(SecFilingGemmaPreprocessorError, match="canonical replay"):
        validate_owned_preprocessing_receipt(changed, **kwargs)


def test_owned_preprocessing_receipt_rejects_wrong_prior_and_provenance() -> None:
    _, kwargs = _owned_preprocessing_case()
    wrong_prior = copy.deepcopy(kwargs)
    wrong_prior["prior_same_form_normalized_source"] = _normalized_source(
        "document-0009.normalized.txt",
        "A different earlier filing.",
    )
    with pytest.raises(SecFilingGemmaPreprocessorError, match="exact normalized text"):
        build_owned_preprocessing_receipt(**wrong_prior)

    changed_prior_text = copy.deepcopy(kwargs)
    changed_prior_text["prior_same_form_normalized_text"] = (
        "A different earlier filing."
    )
    changed_prior_text["prior_same_form_normalized_source"] = _normalized_source(
        "document-0009.normalized.txt",
        changed_prior_text["prior_same_form_normalized_text"],
    )
    with pytest.raises(SecFilingGemmaPreprocessorError, match="canonical replay"):
        build_owned_preprocessing_receipt(**changed_prior_text)

    same_source = copy.deepcopy(kwargs)
    same_source["prior_same_form_normalized_source"]["relative_path"] = (
        "document-0002.normalized.txt"
    )
    with pytest.raises(SecFilingGemmaPreprocessorError, match="distinct document"):
        build_owned_preprocessing_receipt(**same_source)

    wrong_kind = copy.deepcopy(kwargs)
    wrong_kind["prior_provenance_kind"] = "development_root_carry_in"
    wrong_kind["prior_same_form_normalized_source"]["relative_path"] = (
        "carry-in-0001.normalized.txt"
    )
    with pytest.raises(
        SecFilingGemmaPreprocessorError,
        match="valid only for intermediate",
    ):
        build_owned_preprocessing_receipt(**wrong_kind)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"sha256": "a" * 64}, "exact normalized text"),
        ({"byte_count": 1}, "exact normalized text"),
        ({"byte_count": True}, "integer from"),
        ({"byte_count": MAX_OWNED_NORMALIZED_SOURCE_BYTES + 1}, "integer from"),
        ({"sha256": "A" * 64}, "bare lowercase"),
        ({"unexpected": "field"}, "contain exactly"),
    ],
)
def test_owned_preprocessing_receipt_rejects_wrong_source_descriptor(
    mutation: dict,
    message: str,
) -> None:
    _, kwargs = _owned_preprocessing_case()
    changed = copy.deepcopy(kwargs)
    changed["current_normalized_source"].update(mutation)

    with pytest.raises(SecFilingGemmaPreprocessorError, match=message):
        build_owned_preprocessing_receipt(**changed)


@pytest.mark.parametrize(
    "relative_path, message",
    [
        ("../document-0002.normalized.txt", "canonical normalized filename"),
        ("document-2.normalized.txt", "canonical normalized filename"),
        ("document-0000.normalized.txt", "ordinal must be positive"),
        ("carry-in-0002.normalized.txt", "canonical document"),
        ("document-0002.normalized.txt/extra", "canonical normalized filename"),
        (r"folder\document-0002.normalized.txt", "canonical normalized filename"),
    ],
)
def test_owned_preprocessing_receipt_rejects_noncanonical_source_path(
    relative_path: str,
    message: str,
) -> None:
    _, kwargs = _owned_preprocessing_case()
    changed = copy.deepcopy(kwargs)
    changed["current_normalized_source"]["relative_path"] = relative_path

    with pytest.raises(SecFilingGemmaPreprocessorError, match=message):
        build_owned_preprocessing_receipt(**changed)


@pytest.mark.parametrize(
    "stage, prior_kind",
    [
        ("intermediate", "development_root_carry_in"),
        ("final", "stage_carry_in"),
    ],
)
def test_stage_receipts_bind_required_carry_reader_and_stage_specific_prior(
    stage: str,
    prior_kind: str,
) -> None:
    current = "Demand improved while liquidity remained stable."
    prior = "Supply risk had increased."
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    kwargs = {
        "scope_kind": "stage_request",
        "scope_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "model_execution_claim_sha256": "3" * 64,
        "sec_reader_receipt_sha256": "4" * 64,
        "carry_in_reader_receipt_sha256": "5" * 64,
        "stage": stage,
        "event_ordinal": 11,
        "accession_number": "0000320193-24-000001",
        "form": "10-K",
        "current_normalized_source": _normalized_source(
            "document-0001.normalized.txt",
            current,
        ),
        "prior_same_form_normalized_source": _normalized_source(
            "carry-in-0001.normalized.txt",
            prior,
        ),
        "prior_provenance_kind": prior_kind,
        "preprocessor_source_sha256": "6" * 64,
        "preprocessed_event": artifact,
        "current_normalized_text": current,
        "prior_same_form_normalized_text": prior,
    }

    receipt = build_owned_preprocessing_receipt(**kwargs)
    assert receipt["carry_in_reader_receipt_sha256"] == "5" * 64
    assert receipt["prior_provenance_kind"] == prior_kind

    missing_carry = copy.deepcopy(kwargs)
    missing_carry["carry_in_reader_receipt_sha256"] = None
    with pytest.raises(SecFilingGemmaPreprocessorError, match="bare lowercase"):
        build_owned_preprocessing_receipt(**missing_carry)


def test_development_first_form_allows_no_prior_but_never_a_carry_reader() -> None:
    current = "Demand improved while liquidity remained stable."
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    kwargs = {
        "scope_kind": "development_root",
        "scope_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "model_execution_claim_sha256": "3" * 64,
        "sec_reader_receipt_sha256": "4" * 64,
        "carry_in_reader_receipt_sha256": None,
        "stage": "development",
        "event_ordinal": 4,
        "accession_number": "0000320193-00-000001",
        "form": "10-K",
        "current_normalized_source": _normalized_source(
            "document-0001.normalized.txt",
            current,
        ),
        "prior_same_form_normalized_source": None,
        "prior_provenance_kind": None,
        "preprocessor_source_sha256": "5" * 64,
        "preprocessed_event": artifact,
        "current_normalized_text": current,
        "prior_same_form_normalized_text": None,
    }
    receipt = build_owned_preprocessing_receipt(**kwargs)
    assert receipt["prior_same_form_normalized_source"] is None
    assert receipt["prior_provenance_kind"] is None

    with_carry = {**kwargs, "carry_in_reader_receipt_sha256": "6" * 64}
    with pytest.raises(SecFilingGemmaPreprocessorError, match="cannot bind a carry"):
        build_owned_preprocessing_receipt(**with_carry)


@pytest.mark.parametrize(
    "scope_kind, stage",
    [
        ("development_root", "intermediate"),
        ("development_root", "final"),
        ("stage_request", "development"),
    ],
)
def test_owned_preprocessing_scope_and_stage_must_match(
    scope_kind: str,
    stage: str,
) -> None:
    _, kwargs = _owned_preprocessing_case()
    changed = copy.deepcopy(kwargs)
    changed["scope_kind"] = scope_kind
    changed["stage"] = stage
    if stage != "development":
        changed["carry_in_reader_receipt_sha256"] = "6" * 64

    with pytest.raises(SecFilingGemmaPreprocessorError, match="inconsistent"):
        build_owned_preprocessing_receipt(**changed)


def test_builds_exact_canonical_payload_report_and_hashes() -> None:
    current = (
        "Apple revenue increased 12% in September. "
        "Results may vary as costs decline and liquidity remains stable."
    )
    prior = "A\u0440ple revenue was twelve billion dollars on Tuesday."

    artifact = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior,
        identity_lexicon=IDENTITY_LEXICON,
    )

    assert artifact["schema_version"] == PREPROCESSED_EVENT_SCHEMA_VERSION
    assert artifact["preprocessor_version"] == PREPROCESSOR_VERSION
    assert artifact["input_scope"] == (
        "exact_current_and_optional_immediate_prior_same_form_only"
    )
    assert artifact["current_filing_sha256"] == hashlib.sha256(
        current.encode("utf-8")
    ).hexdigest()
    assert artifact["prior_same_form_filing_sha256"] == hashlib.sha256(
        prior.encode("utf-8")
    ).hexdigest()
    normalized_lexicon = sorted(term.strip().casefold() for term in IDENTITY_LEXICON)
    assert artifact["identity_lexicon_sha256"] == canonical_sha256(normalized_lexicon)

    ids = [sentence["id"] for sentence in artifact["sentences"]]
    assert ids == ["C0001", "C0002", "P0001"]
    assert artifact["model_payload"] == build_extractor_model_payload(
        artifact["sentences"]
    )
    assert artifact["model_payload_sha256"] == canonical_sha256(
        artifact["model_payload"]
    )
    assert artifact["sentences_sha256"] == canonical_sha256(artifact["sentences"])

    texts = _texts(artifact)
    expected_bytes = len("\n".join(texts).encode("utf-8"))
    assert artifact["redaction_report"] == {
        "identity_matches_remaining": 0,
        "numeric_matches_remaining": 0,
        "date_matches_remaining": 0,
        "forbidden_context_fields": 0,
        "sentence_count": len(texts),
        "utf8_bytes": expected_bytes,
        "maximum_sentence_characters": max(map(len, texts)),
    }
    body = {
        key: value
        for key, value in artifact.items()
        if key != "preprocessed_event_sha256"
    }
    assert artifact["preprocessed_event_sha256"] == canonical_sha256(body)
    assert validate_preprocessed_event(
        artifact,
        current_normalized_text=current,
        prior_same_form_normalized_text=prior,
        identity_lexicon=IDENTITY_LEXICON,
    ) == artifact["preprocessed_event_sha256"]

    # The exact model-user JSON contains only sentence ids and safe text.
    user_content = json.loads(artifact["model_payload"]["messages"][1]["content"])
    assert user_content == {"sentences": artifact["sentences"]}


def test_unicode_identity_digits_currency_numbers_and_dates_are_redacted() -> None:
    current = (
        "A\u0440\u0440le and \uff21\uff50\uff50\uff4c\uff45 reported \uff11\uff12\uff13 dollars and \u0661\u0662\u0663 euros. "
        "\u0399Phone demand rose 25\uff05, while Tim Cook discussed twelve billion shares. "
        "The period ended September thirtieth, 2024, on Tuesday. "
        "Results may vary and costs may decline. "
        "Revenue improved in May. May results were stronger. May improve is modal."
    )
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    output = " ".join(_texts(artifact))
    lowered = output.casefold()

    assert output.isascii()
    assert not any(character.isdigit() for character in output)
    for leaked in (
        "apple",
        "iphone",
        "tim cook",
        "september",
        "thirtieth",
        "tuesday",
        "dollars",
        "euros",
        "twelve",
        "billion",
    ):
        assert leaked not in lowered
    assert "[IDENTITY]" in output
    assert "[MONEY]" in output
    assert "[RATE]" in output
    assert "[DATE]" in output
    assert "Results may vary" in output
    assert "costs may decline" in output
    assert "May improve is modal" in output
    assert "in May" not in output
    assert "May results" not in output
    assert artifact["redaction_report"]["identity_matches_remaining"] == 0
    assert artifact["redaction_report"]["numeric_matches_remaining"] == 0
    assert artifact["redaction_report"]["date_matches_remaining"] == 0


def test_identity_redaction_resists_separators_format_controls_and_unmapped_scripts() -> None:
    current = (
        "A-p-p-l-e demand improved. "
        "A p p l e liquidity remained stable. "
        "A\u200bp\u2060p\u200dl\ufeffe costs declined. "
        "A. p. p. l. e margins improved. "
        "A\np\np\nl\ne supply pressure declined. "
        "i-p-h-o-n-e sales increased. "
        "T-i-m C-o-o-k discussed supply risk. "
        "App\u04c0e demand weakened in region alpha. "
        "Ap\u13e2le margins improved in region beta. "
        "I\u0d20hone inventory declined in region gamma. "
        "Mac\u05d0book costs increased in region delta. "
        "Results may vary as demand may improve and costs may decline."
    )
    first = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    output = " ".join(_texts(first))
    lowered = output.casefold()

    assert replay == first
    assert output.count("[IDENTITY]") >= 7
    assert output.count("[UNICODE]") >= 4
    for leaked_fragment in (
        "a-p-p-l-e",
        "a p p l e",
        "a. p. p. l. e",
        "iphone",
        "t-i-m",
        "app e",
        "ap le",
        "i hone",
        "mac book",
    ):
        assert leaked_fragment not in lowered
    assert "Results may vary as demand may improve and costs may decline" in output
    assert validate_preprocessed_event(
        first,
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    ) == first["preprocessed_event_sha256"]


def test_artifact_fields_are_accepted_by_existing_extractor_request_validator() -> None:
    # Reuse the contract suite's complete synthetic universe so this test proves
    # compatibility with the real validator rather than merely duplicating its
    # surface shape here.
    from tests import test_sec_filing_gemma_contract as contract_scaffold

    universe = contract_scaffold._universe()
    current = next(
        record
        for record in universe["records"]
        if record["form"] == "10-Q"
        and record["availability_session"].startswith("2001-")
    )
    prior = [
        record
        for record in universe["records"]
        if record["form"] == current["form"]
        and (record["availability_session"], record["accession_number"])
        < (current["availability_session"], current["accession_number"])
    ][-1]
    current_text = "A-p-p-l-e demand improved while costs declined."
    prior_text = "A\u200bp\u2060p\u200dl\ufeffe demand had weakened."
    artifact = preprocess_filing_event(
        current_normalized_text=current_text,
        prior_same_form_normalized_text=prior_text,
        identity_lexicon=contract_scaffold.IDENTITY_TERMS,
    )

    base_content = contract_scaffold._stage_content(universe, "development")
    documents = [
        {
            key: document[key]
            for key in (
                "accession_number",
                "primary_document_sha256",
                "normalized_text_sha256",
                "primary_document_bytes",
                "normalized_text_bytes",
            )
        }
        for document in copy.deepcopy(base_content["documents"])
    ]
    for document in documents:
        if document["accession_number"] == current["accession_number"]:
            document["normalized_text_sha256"] = artifact["current_filing_sha256"]
            document["normalized_text_bytes"] = len(current_text.encode("utf-8"))
        elif document["accession_number"] == prior["accession_number"]:
            document["normalized_text_sha256"] = artifact[
                "prior_same_form_filing_sha256"
            ]
            document["normalized_text_bytes"] = len(prior_text.encode("utf-8"))
    content = build_stage_content_manifest(
        artifact_stage="development",
        corpus_universe_sha256=universe["universe_sha256"],
        documents=documents,
        universe_manifest=universe,
    )
    redacted_manifest = build_redacted_input_manifest(
        artifact_stage="development",
        accession_number=current["accession_number"],
        corpus_universe_sha256=universe["universe_sha256"],
        model_payload_sha256=artifact["model_payload_sha256"],
        preprocessed_event_sha256=artifact["preprocessed_event_sha256"],
        owned_preprocessing_receipt_sha256="2" * 64,
        sec_reader_receipt_sha256="3" * 64,
        carry_in_reader_receipt_sha256=None,
        universe_manifest=universe,
        stage_content_manifest=content,
    )
    request = {
        "request_version": EXTRACTOR_REQUEST_VERSION,
        "preprocessor_version": artifact["preprocessor_version"],
        "corpus_universe_sha256": universe["universe_sha256"],
        "identity_lexicon_sha256": artifact["identity_lexicon_sha256"],
        "redacted_input_manifest_sha256": redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        "stage": "development",
        "current_accession_number": current["accession_number"],
        "current_form": current["form"],
        "current_availability_session": current["availability_session"],
        "current_filing_sha256": artifact["current_filing_sha256"],
        "prior_accession_number": prior["accession_number"],
        "prior_availability_session": prior["availability_session"],
        "prior_same_form_filing_sha256": artifact[
            "prior_same_form_filing_sha256"
        ],
        "model_payload": artifact["model_payload"],
        "model_payload_sha256": artifact["model_payload_sha256"],
        "redaction_report": artifact["redaction_report"],
    }
    candidate = contract_scaffold._candidate(universe)

    validated = validate_extractor_request(
        request,
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        universe_manifest=universe,
        content_manifests_by_stage={"development": content},
        expected_content_manifest_sha256s={
            "development": content["content_manifest_sha256"]
        },
        session_dates=contract_scaffold.SESSIONS,
        forbidden_identity_terms=contract_scaffold.IDENTITY_TERMS,
        redacted_input_manifest=redacted_manifest,
        expected_redacted_input_manifest_sha256=redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        expected_preprocessed_event_sha256=artifact[
            "preprocessed_event_sha256"
        ],
        expected_owned_preprocessing_receipt_sha256="2" * 64,
        expected_sec_reader_receipt_sha256="3" * 64,
        expected_carry_in_reader_receipt_sha256=None,
    )
    assert validated["model_payload_sha256"] == artifact["model_payload_sha256"]
    assert validated["sentence_ids"] == tuple(
        sentence["id"] for sentence in artifact["sentences"]
    )


def test_market_outcome_benchmark_and_trading_context_is_never_selected() -> None:
    current = (
        "Apple demand improved while operating costs declined. "
        "The stock price return beat the market. "
        "An analyst forecast gave a price target. "
        "A trading signal recommended a long position. "
        "Benchmark performance was favorable. "
        "Liquidity remained stable and supply risk declined."
    )
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    output = " ".join(_texts(artifact)).casefold()

    assert "demand improved" in output
    assert "liquidity remained stable" in output
    for forbidden in (
        "stock price",
        "beat the market",
        "analyst forecast",
        "price target",
        "trading signal",
        "long position",
        "benchmark performance",
    ):
        assert forbidden not in output
    assert artifact["redaction_report"]["forbidden_context_fields"] == 0


def test_fixed_independent_caps_ids_character_limit_and_byte_limit() -> None:
    current_sentences = [
        (
            f"Demand improved for segment {_alpha_suffix(index)} while cost pressure "
            "and supply risk remained uncertain "
            + "operational narrative " * 24
        )
        for index in range(90)
    ]
    prior_sentences = [
        (
            f"Liquidity weakened for region {_alpha_suffix(index)} while management "
            "described regulatory risk "
            + "historical narrative " * 24
        )
        for index in range(90)
    ]
    artifact = preprocess_filing_event(
        current_normalized_text=". ".join(current_sentences) + ".",
        prior_same_form_normalized_text=". ".join(prior_sentences) + ".",
        identity_lexicon=IDENTITY_LEXICON,
    )

    current_ids = [
        sentence["id"]
        for sentence in artifact["sentences"]
        if sentence["id"].startswith("C")
    ]
    prior_ids = [
        sentence["id"]
        for sentence in artifact["sentences"]
        if sentence["id"].startswith("P")
    ]
    assert len(current_ids) == MAX_CURRENT_SENTENCES
    assert len(prior_ids) == MAX_PRIOR_SENTENCES
    assert len(artifact["sentences"]) == MAX_SENTENCES
    assert current_ids == [
        f"C{index:04d}" for index in range(1, MAX_CURRENT_SENTENCES + 1)
    ]
    assert prior_ids == [
        f"P{index:04d}" for index in range(1, MAX_PRIOR_SENTENCES + 1)
    ]
    assert all(
        1 <= len(sentence["text"]) <= MAX_SENTENCE_CHARACTERS
        for sentence in artifact["sentences"]
    )
    assert (
        artifact["redaction_report"]["maximum_sentence_characters"]
        <= MAX_SENTENCE_CHARACTERS
    )
    assert artifact["redaction_report"]["utf8_bytes"] <= MAX_INPUT_BYTES


def test_empty_and_thin_documents_are_deterministic_and_validator_compatible() -> None:
    empty = preprocess_filing_event(
        current_normalized_text="",
        identity_lexicon=IDENTITY_LEXICON,
    )
    assert [sentence["id"] for sentence in empty["sentences"]] == ["C0001"]
    assert _texts(empty) == [
        "No usable current business or risk sentence remained after redaction."
    ]
    assert empty["prior_same_form_filing_sha256"] is None

    thin = preprocess_filing_event(
        current_normalized_text="Narrative remained concise.",
        prior_same_form_normalized_text="",
        identity_lexicon=IDENTITY_LEXICON,
    )
    assert _texts(thin, "C") == ["Narrative remained concise"]
    assert _texts(thin, "P") == [
        "No usable prior business or risk sentence remained after redaction."
    ]
    assert thin["prior_same_form_filing_sha256"] == hashlib.sha256(b"").hexdigest()
    assert thin["model_payload"] == build_extractor_model_payload(thin["sentences"])


def test_pair_and_stage_prefix_invariance_and_no_hidden_state() -> None:
    current = ". ".join(
        f"Demand improved in segment {_alpha_suffix(index)}"
        for index in range(60)
    )
    other_current = "Liquidity weakened while costs increased."
    prior_a = "Demand had weakened while inventory risk increased."
    prior_b = "Pricing remained stable while supply pressure declined."

    current_only = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    paired_a = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior_a,
        identity_lexicon=IDENTITY_LEXICON,
    )
    paired_b = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior_b,
        identity_lexicon=IDENTITY_LEXICON,
    )
    different_current_same_prior = preprocess_filing_event(
        current_normalized_text=other_current,
        prior_same_form_normalized_text=prior_a,
        identity_lexicon=IDENTITY_LEXICON,
    )

    assert [
        sentence for sentence in current_only["sentences"] if sentence["id"].startswith("C")
    ] == [sentence for sentence in paired_a["sentences"] if sentence["id"].startswith("C")]
    assert [sentence for sentence in paired_a["sentences"] if sentence["id"].startswith("C")] == [
        sentence for sentence in paired_b["sentences"] if sentence["id"].startswith("C")
    ]
    assert [sentence for sentence in paired_a["sentences"] if sentence["id"].startswith("P")] == [
        sentence
        for sentence in different_current_same_prior["sentences"]
        if sentence["id"].startswith("P")
    ]

    # Interleaving a completely unrelated event cannot alter a replay because
    # there is no global cache, corpus input, sequence counter, or random state.
    first = copy.deepcopy(paired_a)
    preprocess_filing_event(
        current_normalized_text="Unrelated future narrative about legal risk.",
        prior_same_form_normalized_text="Another unrelated narrative.",
        identity_lexicon=IDENTITY_LEXICON,
    )
    replay = preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior_a,
        identity_lexicon=IDENTITY_LEXICON,
    )
    assert replay == first

    with pytest.raises(TypeError, match="unexpected keyword"):
        preprocess_filing_event(
            current_normalized_text=current,
            prior_same_form_normalized_text=prior_a,
            identity_lexicon=IDENTITY_LEXICON,
            future_normalized_text="Not an authorized input.",  # type: ignore[call-arg]
        )


def test_exact_source_bytes_are_hashed_even_when_safe_text_is_equivalent() -> None:
    composed = preprocess_filing_event(
        current_normalized_text="Demand at the cafe improved.",
        identity_lexicon=IDENTITY_LEXICON,
    )
    decomposed = preprocess_filing_event(
        current_normalized_text="Demand at the cafe\u0301 improved.",
        identity_lexicon=IDENTITY_LEXICON,
    )

    assert _texts(composed) == _texts(decomposed)
    assert composed["current_filing_sha256"] != decomposed["current_filing_sha256"]
    assert composed["model_payload_sha256"] == decomposed["model_payload_sha256"]


def test_replay_validator_rejects_any_artifact_or_input_mutation() -> None:
    current = "Demand improved while liquidity remained stable."
    artifact = preprocess_filing_event(
        current_normalized_text=current,
        identity_lexicon=IDENTITY_LEXICON,
    )
    changed = copy.deepcopy(artifact)
    changed["sentences"][0]["text"] = "Demand weakened."

    with pytest.raises(SecFilingGemmaPreprocessorError, match="canonical replay"):
        validate_preprocessed_event(
            changed,
            current_normalized_text=current,
            identity_lexicon=IDENTITY_LEXICON,
        )
    with pytest.raises(SecFilingGemmaPreprocessorError, match="canonical replay"):
        validate_preprocessed_event(
            artifact,
            current_normalized_text=current + " Extra text.",
            identity_lexicon=IDENTITY_LEXICON,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"current_normalized_text": b"not text", "identity_lexicon": IDENTITY_LEXICON},
            "current_normalized_text must be a string",
        ),
        (
            {
                "current_normalized_text": "Demand improved.",
                "prior_same_form_normalized_text": b"not text",
                "identity_lexicon": IDENTITY_LEXICON,
            },
            "prior_same_form_normalized_text must be a string",
        ),
        (
            {"current_normalized_text": "Demand improved.", "identity_lexicon": ("apple",)},
            "omits a mandatory",
        ),
        (
            {
                "current_normalized_text": "Demand improved.",
                "identity_lexicon": IDENTITY_LEXICON + (" Apple ",),
            },
            "unique normalized",
        ),
    ],
)
def test_invalid_inputs_fail_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(SecFilingGemmaPreprocessorError, match=message):
        preprocess_filing_event(**kwargs)
