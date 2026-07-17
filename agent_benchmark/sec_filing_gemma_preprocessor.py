"""Pure per-event sentence preprocessing for the SEC-filing Gemma experiment.

The public API deliberately accepts no accession, stage, availability date,
market data, outcome, label, or trading state.  A caller can provide only the
exact normalized text for one current filing, the optional exact normalized
text for its immediate prior same-form filing, and the frozen identity
lexicon.  This narrow interface is the isolation boundary: processing another
event, a longer stage prefix, or a future filing cannot affect this event.

The implementation performs no I/O and reads no clock, environment, network,
filesystem, random generator, or mutable cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import html
import json
import re
from types import MappingProxyType
from typing import Any, Final
import unicodedata

from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_EXECUTIVE_IDENTITY_TERMS,
    CANONICAL_IDENTITY_LEXICON,
    CANONICAL_IDENTITY_LEXICON_SHA256,
    MANDATORY_IDENTITY_TERMS,
    MAX_INPUT_BYTES,
    MAX_SENTENCE_CHARACTERS,
    MAX_SENTENCES,
    PREPROCESSOR_VERSION,
    _SPELLED_ABSOLUTE_TERMS,
    _SPELLED_DATE_TERMS,
    build_extractor_model_payload,
    canonical_sha256,
)


PREPROCESSED_EVENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-preprocessed-event-v1"
)
OWNED_PREPROCESSING_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-preprocessing-receipt-v1"
)

# Owned SEC batches are sealed with the same per-file ceiling.  Repeating the
# fixed value here keeps this pure module independent of the effectful reveal
# store (which is expected to import this module, never the reverse).
MAX_OWNED_NORMALIZED_SOURCE_BYTES: Final[int] = 128 * 1024 * 1024
_MAX_OWNED_PREPROCESSED_EVENT_CANONICAL_BYTES: Final[int] = 1024 * 1024
_MAX_OWNED_PREPROCESSED_EVENT_NODES: Final[int] = 8192
_MAX_OWNED_PREPROCESSED_EVENT_DEPTH: Final[int] = 16

OWNED_PREPROCESSING_SCOPE_KINDS: Final[frozenset[str]] = frozenset(
    {"development_root", "stage_request"}
)
OWNED_PREPROCESSING_PRIOR_PROVENANCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "same_scope_document",
        "development_root_carry_in",
        "stage_carry_in",
    }
)

# A fixed independent allocation is intentional.  The selected C sentences
# cannot change when a prior filing is added or changed, and the selected P
# sentences cannot change when the current filing changes.  Leaving unused
# capacity unfilled is preferable to making one side depend on the other.
MAX_CURRENT_SENTENCES: Final[int] = MAX_SENTENCES // 2
MAX_PRIOR_SENTENCES: Final[int] = MAX_SENTENCES - MAX_CURRENT_SENTENCES

_EMPTY_CURRENT_SENTENCE: Final[str] = (
    "No usable current business or risk sentence remained after redaction."
)
_EMPTY_PRIOR_SENTENCE: Final[str] = (
    "No usable prior business or risk sentence remained after redaction."
)

_UNAMBIGUOUS_CALENDAR_WORDS: Final[frozenset[str]] = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
)
_DATE_ORDINAL_WORDS: Final[frozenset[str]] = frozenset(
    _SPELLED_DATE_TERMS - _UNAMBIGUOUS_CALENDAR_WORDS - {"may"}
)
_MAY_TEMPORAL_PREPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "after",
        "before",
        "by",
        "during",
        "from",
        "in",
        "on",
        "since",
        "through",
        "until",
    }
)
_MAY_CALENDAR_FOLLOWERS: Final[frozenset[str]] = frozenset(
    {
        "month",
        "period",
        "quarter",
        "revenue",
        "results",
        "sales",
        "was",
    }
)

_CURRENCY_WORDS: Final[frozenset[str]] = frozenset(
    {
        "cad",
        "chf",
        "cny",
        "dollar",
        "dollars",
        "eur",
        "euro",
        "euros",
        "gbp",
        "jpy",
        "pound",
        "pounds",
        "rmb",
        "usd",
        "yen",
    }
)
_RATE_WORDS: Final[frozenset[str]] = frozenset({"percent", "percentage"})

# Unicode NFKC handles full-width Latin forms.  These explicit mappings cover
# common Greek/Cyrillic/letterlike homoglyphs before unsupported Unicode is
# removed, so strings such as A<Cyrillic er>ple cannot evade the lexicon.
_CONFUSABLE_TRANSLATION: Final[Mapping[int, str]] = MappingProxyType(
    str.maketrans(
        {
            # Cyrillic
            "А": "A",
            "а": "a",
            "В": "B",
            "в": "b",
            "С": "C",
            "с": "c",
            "Е": "E",
            "е": "e",
            "Н": "H",
            "һ": "h",
            "І": "I",
            "і": "i",
            "Ј": "J",
            "ј": "j",
            "К": "K",
            "к": "k",
            "М": "M",
            "м": "m",
            "О": "O",
            "о": "o",
            "Р": "P",
            "р": "p",
            "Ѕ": "S",
            "ѕ": "s",
            "Т": "T",
            "т": "t",
            "Х": "X",
            "х": "x",
            "У": "Y",
            "у": "y",
            "ԁ": "d",
            "ԛ": "q",
            "ԝ": "w",
            # Greek
            "Α": "A",
            "α": "a",
            "Β": "B",
            "β": "b",
            "Ε": "E",
            "ε": "e",
            "Η": "H",
            "Ι": "I",
            "ι": "i",
            "Κ": "K",
            "κ": "k",
            "Μ": "M",
            "Ν": "N",
            "ν": "v",
            "Ο": "O",
            "ο": "o",
            "Ρ": "P",
            "ρ": "p",
            "Τ": "T",
            "τ": "t",
            "Χ": "X",
            "χ": "x",
            "Υ": "Y",
            "υ": "y",
            "Ζ": "Z",
            "ζ": "z",
            # Letterlike characters not always reduced as desired by NFKC.
            "ɡ": "g",
            "ı": "i",
            "ſ": "s",
        }
    )
)

_PUNCTUATION_TRANSLATION: Final[Mapping[int, str]] = MappingProxyType(
    str.maketrans(
        {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2022": ". ",
            "\u2026": "...",
            "\u2044": "/",
            "\u2212": "-",
            "\uff05": "%",
            "\u066a": "%",
        }
    )
)

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+")
_BARE_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_AAPL_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z"
)
_NORMALIZED_SOURCE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<kind>document|carry-in)-(?P<ordinal>[0-9]{4})\.normalized\.txt\Z"
)
_CAPITALIZED_PERSON_TOKEN_PATTERN: Final[str] = (
    r"(?:[A-Z]\.|[A-Z][A-Za-z'-]{1,63})"
)
_LOWERCASE_PERSON_PARTICLE_PATTERN: Final[str] = (
    r"(?:al|bin|da|de|del|den|der|di|du|la|le|van|von)"
)
_CAPITALIZED_PERSON_NAME_PATTERN: Final[str] = (
    rf"{_CAPITALIZED_PERSON_TOKEN_PATTERN}"
    rf"(?:\s+{_CAPITALIZED_PERSON_TOKEN_PATTERN}"
    rf"(?:\s+{_CAPITALIZED_PERSON_TOKEN_PATTERN})?"
    rf"|\s+{_LOWERCASE_PERSON_PARTICLE_PATTERN}"
    rf"(?:\s+{_LOWERCASE_PERSON_PARTICLE_PATTERN})?"
    rf"\s+{_CAPITALIZED_PERSON_TOKEN_PATTERN})?"
)
_SPEECH_VERB_PATTERN: Final[str] = (
    r"(?:said|stated|noted|added|announced|explained|commented|reported|indicated)"
)
_LEADERSHIP_TITLE_PATTERN: Final[str] = (
    r"(?:CEO|CFO|COO|CTO|CIO|CMO|"
    r"chief\s+(?:(?:[A-Za-z]+\s+){0,2}officer|executive|financial|"
    r"operating|technology|information|marketing|legal|people|accounting|"
    r"strategy|revenue|commercial|communications|design|security|scientist|"
    r"architect)|"
    r"(?:senior\s+|executive\s+|independent\s+|non-executive\s+)?director|"
    r"(?:board\s+|co-)?chair(?:man|woman|person)?|"
    r"(?:senior\s+|executive\s+)?vice\s+president|president|"
    r"general\s+counsel|treasurer|controller|secretary)"
)
_LEADERSHIP_TITLE_PERSON_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?P<title>(?<![A-Za-z])(?i:{_LEADERSHIP_TITLE_PATTERN})(?![A-Za-z]))"
    rf"\s+(?!(?i:officer)\b)(?P<name>{_CAPITALIZED_PERSON_NAME_PATTERN})"
    r"(?![A-Za-z'-])"
)
_PERSON_TITLE_APPOSITION_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![A-Za-z])(?P<name>{_CAPITALIZED_PERSON_NAME_PATTERN})"
    r"(?![A-Za-z'-])(?P<before_title>\s*,\s*)"
    rf"(?P<title>(?i:{_LEADERSHIP_TITLE_PATTERN})(?![A-Za-z]))"
    r"(?P<after_title>\s*,)"
)
_PERSON_PARENTHETICAL_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![A-Za-z])(?P<name>{_CAPITALIZED_PERSON_NAME_PATTERN})"
    r"(?![A-Za-z'-])(?P<before_title>\s*\(\s*)"
    rf"(?P<title>(?i:{_LEADERSHIP_TITLE_PATTERN})(?![A-Za-z]))"
    r"(?P<after_title>\s*\))"
)
_APPOINTMENT_TRANSITION_PATTERN: Final[str] = (
    r"(?:(?:was|is|has\s+been|had\s+been)\s+"
    r"(?:appointed|named|elected)|became|serves\s+as|will\s+serve\s+as|"
    r"joined\s+as|named|elected)"
)
_PERSON_APPOINTMENT_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![A-Za-z])(?P<name>{_CAPITALIZED_PERSON_NAME_PATTERN})"
    r"(?![A-Za-z'-])"
    rf"(?P<transition>\s+(?i:{_APPOINTMENT_TRANSITION_PATTERN})\s+)"
    rf"(?P<title>(?i:{_LEADERSHIP_TITLE_PATTERN})(?![A-Za-z]))"
)
_PERSON_BEFORE_SPEECH_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![A-Za-z])(?P<name>{_CAPITALIZED_PERSON_NAME_PATTERN})"
    rf"(?=\s+(?i:{_SPEECH_VERB_PATTERN})\b)"
)
_HONORIFIC_PERSON_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z])(?i:mrs|mr|ms|dr)\.?\s+"
    rf"{_CAPITALIZED_PERSON_NAME_PATTERN}"
    r"(?![A-Za-z'-])"
)
_RAW_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[.!?]+(?=\s|$)|[\r\n]+|;(?=\s|$))"
)
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]*>")


def _word_pattern(words: Sequence[str]) -> re.Pattern[str]:
    choices = "|".join(
        re.escape(word) for word in sorted(words, key=lambda item: (-len(item), item))
    )
    return re.compile(rf"(?<![A-Za-z])(?:{choices})(?![A-Za-z])", re.IGNORECASE)


_DATE_WORD_RE: Final[re.Pattern[str]] = _word_pattern(
    tuple(_UNAMBIGUOUS_CALENDAR_WORDS | _DATE_ORDINAL_WORDS)
)
_CURRENCY_WORD_RE: Final[re.Pattern[str]] = _word_pattern(tuple(_CURRENCY_WORDS))
_RATE_WORD_RE: Final[re.Pattern[str]] = _word_pattern(tuple(_RATE_WORDS))
_ABSOLUTE_WORD_RE: Final[re.Pattern[str]] = _word_pattern(
    tuple(_SPELLED_ABSOLUTE_TERMS - _CURRENCY_WORDS - _RATE_WORDS)
)

_NUMERIC_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9]{1,4}[-/][0-9]{1,2}(?:[-/][0-9]{1,4})?)(?![A-Za-z0-9])"
)
_NUMERIC_MONEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[$]\s*[0-9][0-9,._()\-]*|[0-9][0-9,._()\-]*\s*[$])"
)
_NUMERIC_RATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[0-9][0-9,._()\-]*\s*%|%\s*[0-9][0-9,._()\-]*)"
)
_ANY_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:[.,:/\-][0-9]+)*")

_FORBIDDEN_MARKET_CONTEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:stock|share|security|market)\s+(?:price|return|performance)\b",
        r"\btotal\s+shareholder\s+return\b",
        r"\b(?:analyst|security|stock)\s+forecast\b",
        r"\bprice\s+(?:forecast|target)\b",
        r"\bbuy[- ]and[- ]hold\b",
        r"\bbeat(?:s|ing)?\s+(?:the\s+)?market\b",
        r"\bbenchmark\s+(?:return|result|performance)\b",
        r"\btrading\s+(?:action|position|return|signal|strategy)\b",
        r"\b(?:long|short|cash)\s+(?:action|exposure|position|signal)\b",
    )
)

# These groups determine only which filing-derived sentences fit into the
# fixed cap.  They contain business/risk concepts, never prices, returns,
# labels, realized outcomes, benchmarks, or trading actions.
_BUSINESS_KEYWORD_GROUPS: Final[tuple[frozenset[str], ...]] = (
    frozenset({"customer", "demand", "order", "revenue", "sale", "sales"}),
    frozenset({"price", "priced", "pricing"}),
    frozenset({"gross", "margin", "margins"}),
    frozenset({"cost", "costs", "expense", "expenses", "operating"}),
    frozenset({"capital", "dividend", "investment", "repurchase"}),
    frozenset({"cash", "credit", "debt", "liquidity"}),
    frozenset({"expect", "expects", "guidance", "outlook"}),
    frozenset({"component", "inventory", "supplier", "supply"}),
    frozenset({"compliance", "legal", "litigation", "regulatory"}),
    frozenset({"executive", "management", "officer", "transition"}),
    frozenset({"control", "impairment", "restructuring", "risk", "uncertain"}),
)
_DIRECTIONAL_WORDS: Final[frozenset[str]] = frozenset(
    {
        "decline",
        "declined",
        "declining",
        "deteriorate",
        "deteriorated",
        "deteriorating",
        "improve",
        "improved",
        "improving",
        "increase",
        "increased",
        "increasing",
        "reduce",
        "reduced",
        "reducing",
        "stable",
        "weaken",
        "weakened",
        "withdraw",
        "withdrawn",
    }
)


class SecFilingGemmaPreprocessorError(ValueError):
    """Raised when an input or derived artifact is not canonical and safe."""


def _bare_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _BARE_SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaPreprocessorError(
            f"{location} must be a bare lowercase SHA-256"
        )
    return value


def _strict_positive_int(value: Any, location: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise SecFilingGemmaPreprocessorError(
            f"{location} must be an integer from 1 through {maximum}"
        )
    return value


def _detach_owned_preprocessed_event(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Take one bounded plain-JSON snapshot of an untrusted event mapping."""

    if not isinstance(value, Mapping):
        raise SecFilingGemmaPreprocessorError(
            "preprocessed_event must be a string-keyed mapping"
        )
    nodes = 0
    estimated_bytes = 0

    def detach(item: Any, *, depth: int, root: bool = False) -> Any:
        nonlocal nodes, estimated_bytes
        nodes += 1
        estimated_bytes += 16
        if (
            nodes > _MAX_OWNED_PREPROCESSED_EVENT_NODES
            or estimated_bytes
            > _MAX_OWNED_PREPROCESSED_EVENT_CANONICAL_BYTES
        ):
            raise SecFilingGemmaPreprocessorError(
                "preprocessed_event exceeds its canonical snapshot bound"
            )
        if depth > _MAX_OWNED_PREPROCESSED_EVENT_DEPTH:
            raise SecFilingGemmaPreprocessorError(
                "preprocessed_event exceeds its canonical nesting bound"
            )
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -(2**63) <= item <= 2**63 - 1:
                raise SecFilingGemmaPreprocessorError(
                    "preprocessed_event integer is outside the canonical bound"
                )
            return item
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise SecFilingGemmaPreprocessorError(
                    "preprocessed_event string is not valid UTF-8"
                ) from exc
            estimated_bytes += len(encoded)
            if estimated_bytes > _MAX_OWNED_PREPROCESSED_EVENT_CANONICAL_BYTES:
                raise SecFilingGemmaPreprocessorError(
                    "preprocessed_event exceeds its canonical snapshot bound"
                )
            return item
        if type(item) is list:
            return [detach(child, depth=depth + 1) for child in item]
        if (root and isinstance(item, Mapping)) or type(item) is dict:
            detached: dict[str, Any] = {}
            for key in item:
                if type(key) is not str or key in detached:
                    raise SecFilingGemmaPreprocessorError(
                        "preprocessed_event must contain unique plain string keys"
                    )
                try:
                    encoded_key = key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise SecFilingGemmaPreprocessorError(
                        "preprocessed_event key is not valid UTF-8"
                    ) from exc
                estimated_bytes += len(encoded_key)
                if (
                    estimated_bytes
                    > _MAX_OWNED_PREPROCESSED_EVENT_CANONICAL_BYTES
                ):
                    raise SecFilingGemmaPreprocessorError(
                        "preprocessed_event exceeds its canonical snapshot bound"
                    )
                detached[key] = detach(item[key], depth=depth + 1)
            return detached
        raise SecFilingGemmaPreprocessorError(
            "preprocessed_event must contain only plain JSON values"
        )

    detached = detach(value, depth=0, root=True)
    if type(detached) is not dict:
        raise SecFilingGemmaPreprocessorError(
            "preprocessed_event must detach to an exact built-in dict"
        )
    try:
        canonical_bytes = json.dumps(
            detached,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SecFilingGemmaPreprocessorError(
            "preprocessed_event is not canonical bounded JSON"
        ) from exc
    if len(canonical_bytes) > _MAX_OWNED_PREPROCESSED_EVENT_CANONICAL_BYTES:
        raise SecFilingGemmaPreprocessorError(
            "preprocessed_event exceeds its canonical encoded-byte bound"
        )
    return detached


def _normalized_source_descriptor(
    value: Mapping[str, Any],
    *,
    location: str,
) -> tuple[dict[str, Any], str, int]:
    if not isinstance(value, Mapping) or not all(
        type(key) is str for key in value
    ):
        raise SecFilingGemmaPreprocessorError(
            f"{location} must be a string-keyed mapping"
        )
    if set(value) != {"relative_path", "byte_count", "sha256"}:
        raise SecFilingGemmaPreprocessorError(
            f"{location} must contain exactly relative_path, byte_count, and sha256"
        )
    relative_path = value["relative_path"]
    if type(relative_path) is not str:
        raise SecFilingGemmaPreprocessorError(
            f"{location}.relative_path must be a canonical normalized filename"
        )
    match = _NORMALIZED_SOURCE_PATH_RE.fullmatch(relative_path)
    if match is None:
        raise SecFilingGemmaPreprocessorError(
            f"{location}.relative_path must be a canonical normalized filename"
        )
    ordinal = int(match.group("ordinal"))
    if not 1 <= ordinal <= 9999:
        raise SecFilingGemmaPreprocessorError(
            f"{location}.relative_path ordinal must be positive"
        )
    byte_count = _strict_positive_int(
        value["byte_count"],
        f"{location}.byte_count",
        maximum=MAX_OWNED_NORMALIZED_SOURCE_BYTES,
    )
    descriptor = {
        "relative_path": relative_path,
        "byte_count": byte_count,
        "sha256": _bare_sha256(value["sha256"], f"{location}.sha256"),
    }
    return descriptor, match.group("kind"), ordinal


def _reconcile_normalized_source(
    descriptor: Mapping[str, Any],
    *,
    normalized_text: str,
    location: str,
) -> None:
    source_hash = _source_sha256(normalized_text, f"{location}_normalized_text")
    byte_count = len(normalized_text.encode("utf-8"))
    if descriptor["sha256"] != source_hash or descriptor["byte_count"] != byte_count:
        raise SecFilingGemmaPreprocessorError(
            f"{location} source descriptor does not match the exact normalized text"
        )


def _source_sha256(text: str, location: str) -> str:
    if not isinstance(text, str):
        raise SecFilingGemmaPreprocessorError(f"{location} must be a string")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SecFilingGemmaPreprocessorError(
            f"{location} must be valid UTF-8 text"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _normalize_lexicon(identity_lexicon: Sequence[str]) -> tuple[str, ...]:
    if isinstance(identity_lexicon, (str, bytes)) or not isinstance(
        identity_lexicon, Sequence
    ):
        raise SecFilingGemmaPreprocessorError(
            "identity_lexicon must be a sequence of strings"
        )
    if any(not isinstance(term, str) or not term.strip() for term in identity_lexicon):
        raise SecFilingGemmaPreprocessorError(
            "identity_lexicon must contain only nonblank strings"
        )
    normalized = tuple(sorted({term.strip().casefold() for term in identity_lexicon}))
    if len(normalized) != len(identity_lexicon):
        raise SecFilingGemmaPreprocessorError(
            "identity_lexicon must contain unique normalized strings"
        )
    if not set(MANDATORY_IDENTITY_TERMS).issubset(normalized):
        raise SecFilingGemmaPreprocessorError(
            "identity_lexicon omits a mandatory issuer identity term"
        )
    return normalized


def _canonical_ascii_source(text: str) -> str:
    value = html.unescape(text)
    value = unicodedata.normalize("NFKC", value)
    # Format controls (zero-width spaces/joiners, bidi controls, BOM, and
    # similar Cf characters) are not semantic separators.  Removing them
    # reconstructs the token they attempted to split before identity matching.
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Cf"
    )
    value = value.translate(_CONFUSABLE_TRANSLATION)
    value = value.translate(_PUNCTUATION_TRANSLATION)

    converted: list[str] = []
    for character in value:
        if unicodedata.category(character) == "Nd":
            converted.append(str(unicodedata.decimal(character)))
        elif unicodedata.category(character) == "Sc":
            converted.append(" [MONEY] ")
        else:
            converted.append(character)
    value = unicodedata.normalize("NFKD", "".join(converted))

    # NFKD makes ordinary accented Latin text safely transliterable and makes
    # combining-mark insertion unable to split an identity.  Any word-like
    # token that still contains an unmapped non-ASCII letter, number, or mark
    # is replaced as a whole.  Replacing the whole token (rather than dropping
    # one character) prevents outputs such as ``App e`` from preserving an
    # inferable fragment of a disguised identity.
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Cf"
        and not unicodedata.combining(character)
    )
    unicode_safe: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"} or character == "_":
            end = index + 1
            while end < len(value):
                next_character = value[end]
                next_category = unicodedata.category(next_character)
                if next_category[0] not in {"L", "M", "N"} and next_character != "_":
                    break
                end += 1
            token = value[index:end]
            if any(
                not token_character.isascii()
                and unicodedata.category(token_character)[0] in {"L", "M", "N"}
                for token_character in token
            ):
                unicode_safe.append("[UNICODE]")
            else:
                unicode_safe.append(token)
            index = end
            continue
        unicode_safe.append(character)
        index += 1
    value = "".join(unicode_safe)

    ascii_characters: list[str] = []
    for character in value:
        if character.isascii():
            ascii_characters.append(character)
        else:
            # A space prevents deletion from joining two formerly separated
            # words into a new token with different redaction semantics.
            ascii_characters.append(" ")
    return _HTML_TAG_RE.sub(" ", "".join(ascii_characters))


def _identity_skeleton(term: str) -> tuple[str, ...]:
    value = _canonical_ascii_source(term).casefold()
    return tuple(re.findall(r"[a-z0-9]+", value))


def _identity_patterns(normalized_lexicon: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    tokenized = {
        tokens
        for term in normalized_lexicon
        if (tokens := _identity_skeleton(term))
    }
    ordered = sorted(
        tokenized,
        key=lambda tokens: (-sum(len(token) for token in tokens), -len(tokens), tokens),
    )
    patterns: list[re.Pattern[str]] = []
    for tokens in ordered:
        flexible_tokens = [
            r"(?:[^A-Za-z0-9]*)".join(re.escape(character) for character in token)
            for token in tokens
        ]
        joined = r"(?:[^A-Za-z0-9]*)".join(flexible_tokens)
        patterns.append(
            re.compile(
                rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
        )
    return tuple(patterns)


def _redact_identities(
    text: str, identity_patterns: Sequence[re.Pattern[str]]
) -> str:
    for pattern in identity_patterns:
        text = pattern.sub("[IDENTITY]", text)
    return text


def _redact_honorific_people(text: str) -> str:
    return _HONORIFIC_PERSON_RE.sub("[IDENTITY]", text)


def _redact_contextual_people(text: str) -> str:
    text = _PERSON_TITLE_APPOSITION_RE.sub(
        lambda match: (
            f"[IDENTITY]{match.group('before_title')}"
            f"{match.group('title')}{match.group('after_title')}"
        ),
        text,
    )
    text = _PERSON_PARENTHETICAL_TITLE_RE.sub(
        lambda match: (
            f"[IDENTITY]{match.group('before_title')}"
            f"{match.group('title')}{match.group('after_title')}"
        ),
        text,
    )
    text = _PERSON_APPOINTMENT_RE.sub(
        lambda match: (
            f"[IDENTITY]{match.group('transition')}{match.group('title')}"
        ),
        text,
    )
    text = _LEADERSHIP_TITLE_PERSON_RE.sub(
        lambda match: f"{match.group('title')} [IDENTITY]",
        text,
    )
    return _PERSON_BEFORE_SPEECH_RE.sub("[IDENTITY]", text)


def _redact_calendar_may(text: str) -> str:
    words = list(_WORD_RE.finditer(text))
    replacements: list[tuple[int, int]] = []
    for index, match in enumerate(words):
        if match.group(0).casefold() != "may":
            continue
        previous_word = words[index - 1].group(0).casefold() if index else None
        next_word = (
            words[index + 1].group(0).casefold()
            if index + 1 < len(words)
            else None
        )
        tail = text[match.end() :]
        followed_by_numeric_date = re.match(r"\s*[,/\-]?\s*[0-9]", tail) is not None
        calendar_use = (
            previous_word in _MAY_TEMPORAL_PREPOSITIONS
            or next_word in _SPELLED_DATE_TERMS
            or next_word in _MAY_CALENDAR_FOLLOWERS
            or followed_by_numeric_date
        )
        if calendar_use:
            replacements.append(match.span())

    for start, end in reversed(replacements):
        text = f"{text[:start]}[DATE]{text[end:]}"
    return text


def _redact_sentence(raw: str, identity_patterns: Sequence[re.Pattern[str]]) -> str:
    text = _WHITESPACE_RE.sub(" ", raw).strip()
    if not text:
        return ""

    text = _redact_contextual_people(text)
    text = _redact_honorific_people(text)
    text = _redact_identities(text, identity_patterns)

    text = _redact_calendar_may(text)
    text = _NUMERIC_DATE_RE.sub("[DATE]", text)
    text = _DATE_WORD_RE.sub("[DATE]", text)
    text = _NUMERIC_MONEY_RE.sub("[MONEY]", text)
    text = _NUMERIC_RATE_RE.sub("[RATE]", text)
    text = _CURRENCY_WORD_RE.sub("[MONEY]", text)
    text = _RATE_WORD_RE.sub("[RATE]", text)
    text = _ABSOLUTE_WORD_RE.sub("[VALUE]", text)
    text = _ANY_NUMBER_RE.sub("[VALUE]", text)
    text = text.replace("$", "[MONEY]").replace("%", "[RATE]")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _contains_forbidden_market_context(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _FORBIDDEN_MARKET_CONTEXT_PATTERNS)


def _chunk_sentence(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    remainder = text.strip()
    while remainder:
        if len(remainder) <= MAX_SENTENCE_CHARACTERS:
            chunks.append(remainder)
            break
        split_at = remainder.rfind(" ", 0, MAX_SENTENCE_CHARACTERS + 1)
        if split_at <= 0:
            split_at = MAX_SENTENCE_CHARACTERS
        chunk = remainder[:split_at].strip()
        remainder = remainder[split_at:].strip()
        if chunk:
            chunks.append(chunk)
    return tuple(chunks)


def _selection_score(text: str) -> int:
    words = frozenset(word.casefold() for word in _WORD_RE.findall(text))
    represented_groups = sum(
        1 for group in _BUSINESS_KEYWORD_GROUPS if words.intersection(group)
    )
    directional = min(4, len(words.intersection(_DIRECTIONAL_WORDS)))
    return represented_groups * 4 + directional


def _select_document_sentences(
    text: str,
    *,
    limit: int,
    identity_patterns: Sequence[re.Pattern[str]],
    empty_sentence: str,
) -> tuple[str, ...]:
    canonical_source = _canonical_ascii_source(text)
    # Redact across the complete document before sentence boundaries are
    # interpreted.  Otherwise ``A. p. p. l. e`` or one-character-per-line
    # insertion could split a forbidden token into separately safe-looking
    # fragments before the flexible identity pattern sees it.
    canonical_source = _redact_contextual_people(canonical_source)
    canonical_source = _redact_honorific_people(canonical_source)
    canonical_source = _redact_identities(canonical_source, identity_patterns)
    raw_sentences = _RAW_SENTENCE_SPLIT_RE.split(canonical_source)
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    source_position = 0
    for raw in raw_sentences:
        redacted = _redact_sentence(raw, identity_patterns)
        if not redacted or _contains_forbidden_market_context(redacted):
            continue
        for chunk in _chunk_sentence(redacted):
            canonical = _WHITESPACE_RE.sub(" ", chunk).strip()
            if not canonical or _contains_forbidden_market_context(canonical):
                continue
            key = canonical.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((source_position, canonical))
            source_position += 1

    if not candidates:
        return (empty_sentence,)

    ranked = sorted(
        candidates,
        key=lambda item: (-_selection_score(item[1]), item[0], item[1]),
    )[:limit]
    return tuple(text for _, text in sorted(ranked, key=lambda item: item[0]))


def _residual_counts(
    sentences: Sequence[Mapping[str, str]],
    identity_patterns: Sequence[re.Pattern[str]],
) -> tuple[int, int, int, int]:
    identity_count = 0
    numeric_count = 0
    date_count = 0
    forbidden_context_count = 0
    for sentence in sentences:
        text = sentence["text"]
        identity_count += sum(
            len(tuple(pattern.finditer(text))) for pattern in identity_patterns
        )
        identity_count += len(tuple(_HONORIFIC_PERSON_RE.finditer(text)))
        identity_count += len(tuple(_LEADERSHIP_TITLE_PERSON_RE.finditer(text)))
        identity_count += len(tuple(_PERSON_TITLE_APPOSITION_RE.finditer(text)))
        identity_count += len(
            tuple(_PERSON_PARENTHETICAL_TITLE_RE.finditer(text))
        )
        identity_count += len(tuple(_PERSON_APPOINTMENT_RE.finditer(text)))
        identity_count += len(tuple(_PERSON_BEFORE_SPEECH_RE.finditer(text)))
        words = [word.casefold() for word in _WORD_RE.findall(text)]
        numeric_count += len(re.findall(r"[0-9$%]", text))
        numeric_count += sum(word in _SPELLED_ABSOLUTE_TERMS for word in words)
        date_count += sum(
            word in _UNAMBIGUOUS_CALENDAR_WORDS or word in _DATE_ORDINAL_WORDS
            for word in words
        )
        date_count += sum(
            1
            for index, word in enumerate(words)
            if word == "may"
            and (
                (index > 0 and words[index - 1] in _MAY_TEMPORAL_PREPOSITIONS)
                or (
                    index + 1 < len(words)
                    and (
                        words[index + 1] in _SPELLED_DATE_TERMS
                        or words[index + 1] in _MAY_CALENDAR_FOLLOWERS
                    )
                )
            )
        )
        forbidden_context_count += sum(
            pattern.search(text) is not None
            for pattern in _FORBIDDEN_MARKET_CONTEXT_PATTERNS
        )
    return identity_count, numeric_count, date_count, forbidden_context_count


def preprocess_filing_event(
    *,
    current_normalized_text: str,
    identity_lexicon: Sequence[str],
    prior_same_form_normalized_text: str | None = None,
) -> dict[str, Any]:
    """Return one canonical, safe extractor artifact for a single filing event.

    ``prior_same_form_normalized_text`` has a deliberately singular name and
    the API accepts no corpus or arbitrary context collection.  Establishing
    that it is truly the immediate prior same-form filing remains the external
    universe/envelope verifier's responsibility.
    """

    current_hash = _source_sha256(current_normalized_text, "current_normalized_text")
    if prior_same_form_normalized_text is not None:
        prior_hash: str | None = _source_sha256(
            prior_same_form_normalized_text,
            "prior_same_form_normalized_text",
        )
    else:
        prior_hash = None

    normalized_lexicon = _normalize_lexicon(identity_lexicon)
    identity_lexicon_sha256 = canonical_sha256(list(normalized_lexicon))
    identity_patterns = _identity_patterns(normalized_lexicon)

    current = _select_document_sentences(
        current_normalized_text,
        limit=MAX_CURRENT_SENTENCES,
        identity_patterns=identity_patterns,
        empty_sentence=_EMPTY_CURRENT_SENTENCE,
    )
    prior = (
        ()
        if prior_same_form_normalized_text is None
        else _select_document_sentences(
            prior_same_form_normalized_text,
            limit=MAX_PRIOR_SENTENCES,
            identity_patterns=identity_patterns,
            empty_sentence=_EMPTY_PRIOR_SENTENCE,
        )
    )
    sentences = [
        {"id": f"C{index:04d}", "text": text}
        for index, text in enumerate(current, start=1)
    ]
    sentences.extend(
        {"id": f"P{index:04d}", "text": text}
        for index, text in enumerate(prior, start=1)
    )

    if len(sentences) > MAX_SENTENCES:
        raise SecFilingGemmaPreprocessorError("sentence cap invariant failed")
    if any(
        not sentence["text"].isascii()
        or not sentence["text"].strip()
        or sentence["text"] != sentence["text"].strip()
        or len(sentence["text"]) > MAX_SENTENCE_CHARACTERS
        for sentence in sentences
    ):
        raise SecFilingGemmaPreprocessorError("canonical sentence invariant failed")

    identity_remaining, numeric_remaining, date_remaining, forbidden_remaining = (
        _residual_counts(sentences, identity_patterns)
    )
    if any(
        (identity_remaining, numeric_remaining, date_remaining, forbidden_remaining)
    ):
        raise SecFilingGemmaPreprocessorError("redaction residual invariant failed")

    texts = [sentence["text"] for sentence in sentences]
    utf8_bytes = len("\n".join(texts).encode("utf-8"))
    if utf8_bytes > MAX_INPUT_BYTES:
        raise SecFilingGemmaPreprocessorError("UTF-8 byte cap invariant failed")
    report = {
        "identity_matches_remaining": identity_remaining,
        "numeric_matches_remaining": numeric_remaining,
        "date_matches_remaining": date_remaining,
        "forbidden_context_fields": forbidden_remaining,
        "sentence_count": len(sentences),
        "utf8_bytes": utf8_bytes,
        "maximum_sentence_characters": max(len(text) for text in texts),
    }
    model_payload = build_extractor_model_payload(sentences)
    model_payload_sha256 = canonical_sha256(model_payload)
    body: dict[str, Any] = {
        "schema_version": PREPROCESSED_EVENT_SCHEMA_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "input_scope": "exact_current_and_optional_immediate_prior_same_form_only",
        "current_filing_sha256": current_hash,
        "prior_same_form_filing_sha256": prior_hash,
        "identity_lexicon_sha256": identity_lexicon_sha256,
        "sentences": sentences,
        "sentences_sha256": canonical_sha256(sentences),
        "model_payload": model_payload,
        "model_payload_sha256": model_payload_sha256,
        "redaction_report": report,
    }
    return {**body, "preprocessed_event_sha256": canonical_sha256(body)}


def validate_preprocessed_event(
    artifact: Mapping[str, Any],
    *,
    current_normalized_text: str,
    identity_lexicon: Sequence[str],
    prior_same_form_normalized_text: str | None = None,
) -> str:
    """Recompute an event from its only authorized inputs and require equality."""

    if not isinstance(artifact, Mapping) or not all(
        isinstance(key, str) for key in artifact
    ):
        raise SecFilingGemmaPreprocessorError("artifact must be a string-keyed mapping")
    expected = preprocess_filing_event(
        current_normalized_text=current_normalized_text,
        prior_same_form_normalized_text=prior_same_form_normalized_text,
        identity_lexicon=identity_lexicon,
    )
    if dict(artifact) != expected:
        raise SecFilingGemmaPreprocessorError(
            "preprocessed event is not the canonical replay of its exact inputs"
        )
    return expected["preprocessed_event_sha256"]


def build_owned_preprocessing_receipt(
    *,
    scope_kind: str,
    scope_sha256: str,
    candidate_sha256: str,
    model_execution_claim_sha256: str,
    sec_reader_receipt_sha256: str,
    carry_in_reader_receipt_sha256: str | None,
    stage: str,
    event_ordinal: int,
    accession_number: str,
    form: str,
    current_normalized_source: Mapping[str, Any],
    prior_same_form_normalized_source: Mapping[str, Any] | None,
    prior_provenance_kind: str | None,
    preprocessor_source_sha256: str,
    preprocessed_event: Mapping[str, Any],
    current_normalized_text: str,
    prior_same_form_normalized_text: str | None = None,
) -> dict[str, Any]:
    """Bind one pure preprocessing result to its owned reader ancestry.

    The exact normalized texts are accepted only so the already-built
    ``preprocessed_event`` and its byte descriptors can be replayed.  They are
    never copied into the receipt.  The caller supplies no path root, I/O
    handle, alternate lexicon, model payload, or redacted-manifest digest.
    """

    event_snapshot = _detach_owned_preprocessed_event(preprocessed_event)

    if (
        type(scope_kind) is not str
        or scope_kind not in OWNED_PREPROCESSING_SCOPE_KINDS
    ):
        raise SecFilingGemmaPreprocessorError(
            "scope_kind must be development_root or stage_request"
        )
    if type(stage) is not str or stage not in {
        "development",
        "intermediate",
        "final",
    }:
        raise SecFilingGemmaPreprocessorError(
            "stage must be development, intermediate, or final"
        )
    if (scope_kind == "development_root") != (stage == "development"):
        raise SecFilingGemmaPreprocessorError(
            "owned preprocessing scope_kind and stage are inconsistent"
        )

    scope_hash = _bare_sha256(scope_sha256, "scope_sha256")
    candidate_hash = _bare_sha256(candidate_sha256, "candidate_sha256")
    model_claim_hash = _bare_sha256(
        model_execution_claim_sha256,
        "model_execution_claim_sha256",
    )
    sec_reader_hash = _bare_sha256(
        sec_reader_receipt_sha256,
        "sec_reader_receipt_sha256",
    )
    if stage == "development":
        if carry_in_reader_receipt_sha256 is not None:
            raise SecFilingGemmaPreprocessorError(
                "development preprocessing cannot bind a carry-in reader receipt"
            )
        carry_reader_hash: str | None = None
    else:
        carry_reader_hash = _bare_sha256(
            carry_in_reader_receipt_sha256,
            "carry_in_reader_receipt_sha256",
        )

    ordinal = _strict_positive_int(
        event_ordinal,
        "event_ordinal",
        maximum=9999,
    )
    if type(accession_number) is not str or _AAPL_ACCESSION_RE.fullmatch(
        accession_number
    ) is None:
        raise SecFilingGemmaPreprocessorError(
            "accession_number must be a canonical AAPL accession"
        )
    if type(form) is not str or form not in {"10-K", "10-Q"}:
        raise SecFilingGemmaPreprocessorError("form must be 10-K or 10-Q")

    current_source, current_kind, _current_file_ordinal = (
        _normalized_source_descriptor(
            current_normalized_source,
            location="current_normalized_source",
        )
    )
    if current_kind != "document":
        raise SecFilingGemmaPreprocessorError(
            "current normalized source must be a canonical document filename"
        )
    _reconcile_normalized_source(
        current_source,
        normalized_text=current_normalized_text,
        location="current",
    )

    if prior_same_form_normalized_text is None:
        if (
            prior_same_form_normalized_source is not None
            or prior_provenance_kind is not None
        ):
            raise SecFilingGemmaPreprocessorError(
                "absent prior text requires absent prior source and provenance"
            )
        prior_source: dict[str, Any] | None = None
    else:
        if prior_same_form_normalized_source is None:
            raise SecFilingGemmaPreprocessorError(
                "present prior text requires a prior normalized source"
            )
        if (
            type(prior_provenance_kind) is not str
            or prior_provenance_kind
            not in OWNED_PREPROCESSING_PRIOR_PROVENANCE_KINDS
        ):
            raise SecFilingGemmaPreprocessorError(
                "present prior text requires a canonical prior provenance kind"
            )
        prior_source, prior_kind, _prior_file_ordinal = (
            _normalized_source_descriptor(
                prior_same_form_normalized_source,
                location="prior_same_form_normalized_source",
            )
        )
        _reconcile_normalized_source(
            prior_source,
            normalized_text=prior_same_form_normalized_text,
            location="prior_same_form",
        )
        if prior_provenance_kind == "same_scope_document":
            if (
                prior_kind != "document"
                or prior_source["relative_path"]
                == current_source["relative_path"]
            ):
                raise SecFilingGemmaPreprocessorError(
                    "same-scope prior must be a distinct document source"
                )
        elif prior_provenance_kind == "development_root_carry_in":
            if stage != "intermediate" or prior_kind != "carry-in":
                raise SecFilingGemmaPreprocessorError(
                    "development-root carry-in prior is valid only for intermediate"
                )
        elif prior_provenance_kind == "stage_carry_in":
            if stage != "final" or prior_kind != "carry-in":
                raise SecFilingGemmaPreprocessorError(
                    "stage carry-in prior is valid only for final"
                )

    event_hash = validate_preprocessed_event(
        event_snapshot,
        current_normalized_text=current_normalized_text,
        prior_same_form_normalized_text=prior_same_form_normalized_text,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )
    if (
        event_snapshot["identity_lexicon_sha256"]
        != CANONICAL_IDENTITY_LEXICON_SHA256
        or event_snapshot["current_filing_sha256"] != current_source["sha256"]
        or (
            None
            if prior_source is None
            else prior_source["sha256"]
        )
        != event_snapshot["prior_same_form_filing_sha256"]
    ):
        raise SecFilingGemmaPreprocessorError(
            "preprocessed event does not reconcile with its owned source descriptors"
        )

    body: dict[str, Any] = {
        "schema_version": OWNED_PREPROCESSING_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "owned_filing_event_preprocessing",
        "scope_kind": scope_kind,
        "scope_sha256": scope_hash,
        "candidate_sha256": candidate_hash,
        "model_execution_claim_sha256": model_claim_hash,
        "sec_reader_receipt_sha256": sec_reader_hash,
        "carry_in_reader_receipt_sha256": carry_reader_hash,
        "stage": stage,
        "event_ordinal": ordinal,
        "accession_number": accession_number,
        "form": form,
        "current_normalized_source": current_source,
        "prior_same_form_normalized_source": prior_source,
        "prior_provenance_kind": prior_provenance_kind,
        "canonical_identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
        "preprocessor_source_sha256": _bare_sha256(
            preprocessor_source_sha256,
            "preprocessor_source_sha256",
        ),
        "preprocessed_event_sha256": event_hash,
        "model_payload_sha256": _bare_sha256(
            event_snapshot["model_payload_sha256"],
            "preprocessed_event.model_payload_sha256",
        ),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_owned_preprocessing_receipt(
    receipt: Mapping[str, Any],
    *,
    scope_kind: str,
    scope_sha256: str,
    candidate_sha256: str,
    model_execution_claim_sha256: str,
    sec_reader_receipt_sha256: str,
    carry_in_reader_receipt_sha256: str | None,
    stage: str,
    event_ordinal: int,
    accession_number: str,
    form: str,
    current_normalized_source: Mapping[str, Any],
    prior_same_form_normalized_source: Mapping[str, Any] | None,
    prior_provenance_kind: str | None,
    preprocessor_source_sha256: str,
    preprocessed_event: Mapping[str, Any],
    current_normalized_text: str,
    prior_same_form_normalized_text: str | None = None,
) -> str:
    """Replay an owned receipt from independently supplied exact ancestry."""

    if not isinstance(receipt, Mapping) or not all(
        type(key) is str for key in receipt
    ):
        raise SecFilingGemmaPreprocessorError(
            "owned preprocessing receipt must be a string-keyed mapping"
        )
    expected = build_owned_preprocessing_receipt(
        scope_kind=scope_kind,
        scope_sha256=scope_sha256,
        candidate_sha256=candidate_sha256,
        model_execution_claim_sha256=model_execution_claim_sha256,
        sec_reader_receipt_sha256=sec_reader_receipt_sha256,
        carry_in_reader_receipt_sha256=carry_in_reader_receipt_sha256,
        stage=stage,
        event_ordinal=event_ordinal,
        accession_number=accession_number,
        form=form,
        current_normalized_source=current_normalized_source,
        prior_same_form_normalized_source=prior_same_form_normalized_source,
        prior_provenance_kind=prior_provenance_kind,
        preprocessor_source_sha256=preprocessor_source_sha256,
        preprocessed_event=preprocessed_event,
        current_normalized_text=current_normalized_text,
        prior_same_form_normalized_text=prior_same_form_normalized_text,
    )
    if dict(receipt) != expected:
        raise SecFilingGemmaPreprocessorError(
            "owned preprocessing receipt is not the canonical replay of its exact ancestry"
        )
    return expected["receipt_sha256"]


__all__ = [
    "CANONICAL_EXECUTIVE_IDENTITY_TERMS",
    "CANONICAL_IDENTITY_LEXICON",
    "CANONICAL_IDENTITY_LEXICON_SHA256",
    "MAX_CURRENT_SENTENCES",
    "MAX_OWNED_NORMALIZED_SOURCE_BYTES",
    "MAX_PRIOR_SENTENCES",
    "OWNED_PREPROCESSING_PRIOR_PROVENANCE_KINDS",
    "OWNED_PREPROCESSING_RECEIPT_SCHEMA_VERSION",
    "OWNED_PREPROCESSING_SCOPE_KINDS",
    "PREPROCESSED_EVENT_SCHEMA_VERSION",
    "SecFilingGemmaPreprocessorError",
    "build_owned_preprocessing_receipt",
    "preprocess_filing_event",
    "validate_owned_preprocessing_receipt",
    "validate_preprocessed_event",
]
