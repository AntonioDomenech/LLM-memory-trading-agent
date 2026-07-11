from __future__ import annotations

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
    MAX_CURRENT_SENTENCES,
    MAX_PRIOR_SENTENCES,
    PREPROCESSED_EVENT_SCHEMA_VERSION,
    SecFilingGemmaPreprocessorError,
    preprocess_filing_event,
    validate_preprocessed_event,
)


IDENTITY_LEXICON = MANDATORY_IDENTITY_TERMS + (
    "luca maestri",
    "steve jobs",
)


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
