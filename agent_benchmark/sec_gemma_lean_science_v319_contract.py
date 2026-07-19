"""Pure, fail-closed contract for the v3.19 SEC-to-science bridge.

This module performs no filesystem, Git, clock, network, SEC, Yahoo, Ollama,
market, model, broker, or real-money I/O.  It contains only immutable public
authority, canonical identity helpers, and validators for evidence gathered by
the I/O-owning v3.19 modules.

The scientific payload is not copied or translated here.  It is projected from
the pinned v2.2 contract through the twelve preregistered keys and must reproduce
the frozen 38,320-byte canonical payload exactly.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final
from urllib.parse import quote


class ContractViolation(ValueError):
    """A redacted, stable rejection raised for a contract mismatch."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


# Public version and Git authority.
CONTRACT_VERSION: Final[str] = "aapl-sec-gemma-lean-science-v3-19"
CONTRACT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-contract-v1"
)
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-lean-science-v3-19"
DEVELOPMENT_COMMAND: Final[str] = "development"

# The successor code base is intentionally independent from both inherited
# evidence authorities below.
BASE_COMMIT: Final[str] = "503826b5ff0f14b0f1c1a11432bdbd7f41712fc2"
BASE_TREE: Final[str] = "ffa89ed5b2fcd41e4930fa9854ebf25b99308d6b"
BASE_PARENT: Final[str] = "f0080b056634866955e7ee73845d509f2bfcf0f2"
PREREGISTRATION_COMMIT: Final[str] = (
    "299a1458f2021be4ea8e38dbb2048b80fac64a18"
)
PREREGISTRATION_TREE: Final[str] = "163a88727f45082d69a0c7b80b134e585bb9dbce"
PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_19.md"
)
PREREGISTRATION_GIT_BLOB_SHA1: Final[str] = (
    "bb206dbeafe958b42e35dcfcc5798a2e4154b4f1"
)
PREREGISTRATION_LITERAL_SHA256: Final[str] = (
    "5ff7f4070008d2dbb5dce40e48e309282803f7ef40c0b62125eebe5dd30e5fe7"
)
PREREGISTRATION_LITERAL_BYTES: Final[int] = 24_009

# Immutable V3.18 document-only preregistration rejection. It is the immediate
# V3.19 base and never implementation or execution authority.
V318_PREREGISTRATION_COMMIT: Final[str] = BASE_PARENT
V318_PREREGISTRATION_TREE: Final[str] = (
    "999181ff11f60596dbeeb17466e9c660472252b9"
)
V318_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_18.md"
)
V318_PREREGISTRATION_GIT_BLOB_SHA1: Final[str] = (
    "507746c8a9a6c07093fe0aee4a19ab1a5e0e74f5"
)
V318_PREREGISTRATION_LITERAL_SHA256: Final[str] = (
    "1873ab7a0b42c0b347344d4c7c58d252f5f29ab4570018b6cfa138274fe03de9"
)
V318_PREREGISTRATION_LITERAL_BYTES: Final[int] = 24_821
V318_REJECTION_COMMIT: Final[str] = BASE_COMMIT
V318_REJECTION_TREE: Final[str] = BASE_TREE
V318_REJECTION_DOCUMENT_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_18_rejection.md"
)
V318_REJECTION_DOCUMENT_GIT_BLOB_SHA1: Final[str] = (
    "ed59c67a9cdcab165cc1440716dd15fbd43e617b"
)
V318_REJECTION_DOCUMENT_LITERAL_SHA256: Final[str] = (
    "21bff88a7da5faa0e9907c2908884a46dd28998db77365e1013270ae8a1e6e95"
)
V318_REJECTION_DOCUMENT_LITERAL_BYTES: Final[int] = 3_740
V318_REJECTION_COMPARISON_PATH: Final[str] = "e/APPROACH_COMPARISON.md"
V318_REJECTION_COMPARISON_GIT_BLOB_SHA1: Final[str] = (
    "6aac9a11428447266a8d64c4c68ca6b626cd3725"
)
V318_REJECTION_COMPARISON_LITERAL_SHA256: Final[str] = (
    "769dec11a0c12f1a454cd0a99a71c63f3addc12e49b222b6201e5a4eb90c21cb"
)
V318_REJECTION_COMPARISON_LITERAL_BYTES: Final[int] = 37_132

# Immutable V3.17 document-only preregistration rejection. It is predecessor
# evidence only and never implementation or execution authority.
V317_PREREGISTRATION_COMMIT: Final[str] = (
    "6df6e07a1bfb5b8cc41b2ae5f793ff43df05599a"
)
V317_PREREGISTRATION_TREE: Final[str] = (
    "357a9f04b527c6fa13d83faf1d6c89afaae51c66"
)
V317_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_17.md"
)
V317_PREREGISTRATION_GIT_BLOB_SHA1: Final[str] = (
    "fc9509edcba502169312d59f31f05a038b4f2d16"
)
V317_PREREGISTRATION_LITERAL_SHA256: Final[str] = (
    "3b7272fc1b4db83f053f534375df8ff1752fa878933dd23cd8718d8b41b26b9d"
)
V317_PREREGISTRATION_LITERAL_BYTES: Final[int] = 21_021
V317_REJECTION_COMMIT: Final[str] = (
    "4a5bedd72c1e718b46c80b357ecc83561771e837"
)
V317_REJECTION_TREE: Final[str] = (
    "f084829c18826a6d37e266de3847bc48a312eb1e"
)
V317_REJECTION_DOCUMENT_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_17_rejection.md"
)
V317_REJECTION_DOCUMENT_GIT_BLOB_SHA1: Final[str] = (
    "41f8eacfbd857cf297155c79f34db475168afcf0"
)
V317_REJECTION_DOCUMENT_LITERAL_SHA256: Final[str] = (
    "f8b07cb6dced73b6c4a30388264798503ccb1f058644133a6b1b3aa4bcfde42b"
)
V317_REJECTION_DOCUMENT_LITERAL_BYTES: Final[int] = 2_601
V317_REJECTION_COMPARISON_PATH: Final[str] = "e/APPROACH_COMPARISON.md"
V317_REJECTION_COMPARISON_GIT_BLOB_SHA1: Final[str] = (
    "58aac6a63b9e33d777d94f7832afbb109c20f969"
)
V317_REJECTION_COMPARISON_LITERAL_SHA256: Final[str] = (
    "594dbaa7b6f4f76bcb1fea96033276fb21e5e17580265b6dd4ea4ecd5c18e03f"
)
V317_REJECTION_COMPARISON_LITERAL_BYTES: Final[int] = 36_142

# Immutable consumed-predecessor evidence.  These values preserve the complete
# V3.16 P/I/F/X failure topology and are never V3.19 execution authority.
V316_PREREGISTRATION_COMMIT: Final[str] = (
    "2cb4abf35b13502953d4f0b502ab6fc5c638be58"
)
V316_PREREGISTRATION_TREE: Final[str] = (
    "bac40934de7bd299590ac18ffd6a912e125b356f"
)
V316_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_16.md"
)
V316_PREREGISTRATION_GIT_BLOB_SHA1: Final[str] = (
    "fac61696f74b15108d43ff0518ec7d3fbb5dce4c"
)
V316_PREREGISTRATION_LITERAL_SHA256: Final[str] = (
    "e4609a7063784323e94637008a2d034c5904f3f2acf113d629f684ac147c93ba"
)
V316_PREREGISTRATION_LITERAL_BYTES: Final[int] = 30_557
V316_IMPLEMENTATION_COMMIT: Final[str] = (
    "14eea6f536e1fc93ea54dbe8fa6b436a3f462cf4"
)
V316_IMPLEMENTATION_TREE: Final[str] = (
    "8b0b6783aeeaa5d6ee7ec0264f79e29b2bc71ca5"
)
V316_PREFLIGHT_FAILURE_COMMIT: Final[str] = (
    "475084cb48ef081a3b10ac35a554d5e8ca408f7b"
)
V316_PREFLIGHT_FAILURE_TREE: Final[str] = (
    "910fa694aede2b27167af6adbbaa2e9c2522bf2c"
)
V316_PREFLIGHT_FAILURE_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_science_v3_16/DEVELOPMENT_PREFLIGHT.json"
)
V316_PREFLIGHT_FAILURE_GIT_BLOB_SHA1: Final[str] = (
    "c58a04277da9b35581de08086bb0c75df927595c"
)
V316_PREFLIGHT_FAILURE_LITERAL_SHA256: Final[str] = (
    "8485edee1943b54f97ff263db0f6318ea8767b79eee12ae3e096dd5316a5d3fc"
)
V316_PREFLIGHT_FAILURE_LITERAL_BYTES: Final[int] = 622
V316_PREFLIGHT_FAILURE_INTERNAL_SHA256: Final[str] = (
    "e770a83bfcf4ebb2afa477e63e67812fe6d1480c08ce258bbf0b5e1c73e5e94e"
)
V316_REJECTION_COMMIT: Final[str] = (
    "7ff2c5a6915ad1684d803be909647f0d040f121f"
)
V316_REJECTION_TREE: Final[str] = (
    "0ba867ba80f71d722e0e1182c30c8a84812706d9"
)
V316_REJECTION_PATH: Final[str] = "e/APPROACH_COMPARISON.md"
V316_REJECTION_GIT_BLOB_SHA1: Final[str] = (
    "bbe16d8fe516ed1d86dd5d2af027afd80145b595"
)
V316_REJECTION_LITERAL_SHA256: Final[str] = (
    "b34df02d6d9eeadaaf77133a966463cc8418266631d11e970ef04145611b9250"
)
V316_REJECTION_LITERAL_BYTES: Final[int] = 35_283

# The consumed V3.10 implementation is source provenance only.  The V3.19
# bridge reuses its exact public source-authority projection, never its state.
V310_IMPLEMENTATION_COMMIT: Final[str] = (
    "c2776bae573faa7963bd31f89f2e570ce704a207"
)
V310_IMPLEMENTATION_TREE: Final[str] = (
    "37ffbd7992ce658a86b6fa917ad843bc80c6d0db"
)
V310_PREREGISTRATION_COMMIT: Final[str] = (
    "bf66bb95f1ab44bb36b98523f6246761c23bcca3"
)
V310_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_10.md"
)

# Immutable v3.8 SEC-source authority.
V38_SOURCE_COMMIT: Final[str] = "0c4b01cf5f1ef77548658d9bfa38e76fb1b70635"
V38_SOURCE_TREE: Final[str] = "68a9176ed22edeb2edc0f00ae1179c8724603539"
V38_SOURCE_PARENT: Final[str] = "a0f971184ad26630478182be97f01c46c80498e1"

# Rejected v3.9 preregistration provenance.  It is a donor identity only and
# never execution authority for v3.19.
V39_INHERITED_PREREG_COMMIT: Final[str] = (
    "d50c33515ed9597b2fc07bb40c922a3f3166fbdd"
)
V39_INHERITED_PREREG_TREE: Final[str] = (
    "946a009fb9e479778ea1881bae302d802578248b"
)
V39_INHERITED_PREREG_PARENT: Final[str] = V38_SOURCE_COMMIT
V39_INHERITED_PREREG_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_9.md"
)
V39_INHERITED_PREREG_GIT_BLOB_SHA1: Final[str] = (
    "2b7042fd948cad91925c9fe9e01784ea0ff739c8"
)
V39_INHERITED_PREREG_LITERAL_SHA256: Final[str] = (
    "feb9bd040e0ed65aed125e52e79e10bc8f87cd7e96776dc67353502fc0983446"
)
V39_INHERITED_PREREG_LITERAL_BYTES: Final[int] = 43_098

# Pushed v3.8 public authority.
V38_TERMINAL_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_8/DEVELOPMENT_ACQUISITION.json"
)
V38_TERMINAL_BLOB_SHA1: Final[str] = (
    "28075db4e7cb85bfcdeb92e7b1aecce28f3a49fa"
)
V38_TERMINAL_LITERAL_SHA256: Final[str] = (
    "c176c6fb5e200dd363be656562d72bc40f722912221fe7f5cd52d8e42cc9723a"
)
V38_TERMINAL_INTERNAL_SHA256: Final[str] = (
    "d72059f39eb4b1019ce83799682eadc7eecf3fd77310f43bee8537433a20cb20"
)
V38_SOURCE_AUTHORITY_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_8/development/"
    "source-authority-d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a.json"
)
V38_SOURCE_AUTHORITY_BLOB_SHA1: Final[str] = (
    "d2d17dc2e1159e79bbcd2ca3f63cb733289b404f"
)
V38_SOURCE_AUTHORITY_LITERAL_SHA256: Final[str] = (
    "d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a"
)
V38_SOURCE_MODULE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_v38_source.py"
)
V38_SOURCE_MODULE_LITERAL_SHA256: Final[str] = (
    "8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04"
)
V38_SOURCE_MODULE_GIT_BLOB_SHA1: Final[str] = (
    "b04263d5b3a8bdd74362fd02bc637d773d76ff9a"
)
V38_ACQUISITION_MODULE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_v38_acquisition.py"
)
V38_ACQUISITION_MODULE_LITERAL_SHA256: Final[str] = (
    "8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6"
)
V38_ACQUISITION_MODULE_GIT_BLOB_SHA1: Final[str] = (
    "b5fbd94ecaff681ac698e47dd9ea0448d7886ee3"
)

# Authenticated, read-only v3.8 private authority.  These are hashes and
# aggregate counts only; no accession, filename, URL, source body, or contact is
# present in this public contract.
V38_CHECKPOINT_FILE_SHA256: Final[str] = (
    "cd2896d627d0694c9efc202c2c8579569265fb4f61f5f74bb9d7db875fbc981e"
)
V38_LOGICAL_CHECKPOINT_SHA256: Final[str] = (
    "704a4a9554444201ec9468e74f8c77abe82a3e3320acce29cac0c7ace11dd6fc"
)
V38_STAGE_SOURCE_SEAL_SHA256: Final[str] = (
    "0eb7c6a83de59d44b3a5ceeca5918b03070ff1f508cc6a5411cd81871368f413"
)
V38_COMPACT_REPLAY_SHA256: Final[str] = (
    "64a7b7776206b008c0dffe26d4a14be9a092281a7834d878332d4f741db05822"
)
V38_ROLE_MANIFEST_INVENTORY_SHA256: Final[str] = (
    "b6c66fba0d7482ee9b1b1892351b0d68dcb43f4ce2a2916e3ea96a316fbc2ad7"
)
V38_ROLE_PLAN_SHA256: Final[str] = (
    "6a03e4f8c2a444dd0612d7b1ff74cc605e026ace54941696a5ae180dd226110a"
)
V38_ROLE_COUNTS: Final[tuple[tuple[str, int], ...]] = (
    ("I", 97),
    ("U", 75),
    ("D", 75),
)
V38_SEC_REQUEST_COUNT: Final[int] = 199
EXPERIMENT_FAMILY_SEC_REQUEST_COUNT: Final[int] = 964
V38_INVENTORY_FILE_COUNT: Final[int] = 1_410
V38_INVENTORY_BYTE_COUNT: Final[int] = 612_601_642
V38_INVENTORY_SHA256: Final[str] = (
    "850f3a022fcaca5b56156d7843f35b05d3bc5732bf480634bd9872bfac7194f2"
)

# Frozen scientific authority.
SCIENTIFIC_PARENT_COMMIT: Final[str] = (
    "efbc481c57e480d48303763163676e64e87df49d"
)
SCIENTIFIC_PARENT_TREE: Final[str] = (
    "bc91d619a4d04680d9600b1acd74f0a29662d0f9"
)
SCIENTIFIC_CONTRACT_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_online_risk_overlay_contract.py"
)
SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1: Final[str] = (
    "8a1c18728eeb7a31864397bb0c8e992b85d2320b"
)
SCIENTIFIC_CONTRACT_LITERAL_SHA256: Final[str] = (
    "fccb45098f505a2f272f970762f928fbe8b75a23380292924b67d2bdd4696d2e"
)
SCIENTIFIC_CONTRACT_INTERNAL_SHA256: Final[str] = (
    "64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d"
)
SCIENCE_PROJECTION_KEYS: Final[tuple[str, ...]] = (
    "objective",
    "event_availability",
    "chronology",
    "data",
    "features",
    "gemma",
    "learner",
    "policy",
    "ledger",
    "metric_definitions",
    "gates",
    "evidence_classification",
)
SCIENCE_PROJECTION_BYTE_COUNT: Final[int] = 38_320
SCIENCE_PROJECTION_SHA256: Final[str] = (
    "1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609"
)
FROZEN_SCIENCE_PROJECTION_SHA256: Final[str] = SCIENCE_PROJECTION_SHA256

# Frozen development boundary and scientific chronology.
DEVELOPMENT_DOCUMENT_COUNT: Final[int] = 75
DEVELOPMENT_FILENAME_PRESENT_COUNT: Final[int] = 73
DEVELOPMENT_FILENAME_MISSING_COUNT: Final[int] = 2
DEVELOPMENT_PILOT_COUNT: Final[int] = 5
DEVELOPMENT_REMAINING_COUNT: Final[int] = 70
DEVELOPMENT_MARKET_START: Final[str] = "1998-01-01"
DEVELOPMENT_MARKET_END_EXCLUSIVE: Final[str] = "2019-01-01"
DEVELOPMENT_CORPUS_END: Final[str] = "2018-12-31"
DEVELOPMENT_WARMUP_END: Final[str] = "2004-12-31"
DEVELOPMENT_BLOCKS: Final[tuple[tuple[str, str, str], ...]] = (
    ("block_1", "2005-01-03", "2007-12-31"),
    ("block_2", "2008-01-02", "2010-12-31"),
    ("block_3", "2011-01-03", "2013-12-31"),
    ("block_4", "2014-01-02", "2016-12-30"),
    ("block_5", "2017-01-03", "2018-12-31"),
)
SOURCE_PROJECTION_ORDER: Final[tuple[str, ...]] = (
    "availability_session",
    "accession",
)
SCIENCE_EVENT_ORDER: Final[tuple[str, ...]] = (
    "availability_session",
    "acceptance_datetime",
    "accession",
)

# V3.19 nullable source/universe authority.  These are schemas and safe
# commitments only; no accession, filename, URL, body, or contact is public.
NULLABLE_SOURCE_RECORD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-nullable-source-record-v1"
)
PRIVATE_SOURCE_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-private-source-identity-v1"
)
COMPATIBILITY_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-compatibility-manifest-v1"
)
NULLABLE_UNIVERSE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-nullable-universe-v1"
)
NULLABLE_UNIVERSE_SEMANTIC_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-nullable-universe-semantic-v1"
)
NULLABLE_CONTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-nullable-content-v1"
)
NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-nullable-universe-event-proof-v1"
)
BLINDED_MODEL_REQUEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-blinded-model-request-v1"
)
PRIMARY_DOCUMENT_BODY_KIND: Final[str] = (
    "selected_embedded_TEXT_bytes_from_authenticated_complete_submission"
)
LEGACY_MISSING_DOCUMENT_IDENTITY: Final[str] = (
    "legacy-sequence-1-no-filename"
)
PRIOR_SELECTION_POLICY: Final[str] = (
    "immediate_predecessor_same_form_by_availability_session_and_accession"
)
SOURCE_AUTHORITY_PINS_SHA256: Final[str] = (
    "31da49f50edde027c3146c7aa8b94f239ee20cc107bd0add91ca6aa55e48080c"
)
CALENDAR_SESSION_COUNT: Final[int] = 6_669
CALENDAR_SESSIONS_SHA256: Final[str] = (
    "e0550f12f98d7e0cf38d6797ae0d0e410bb3f9006793830ed75e0f753ccc5cf9"
)
COMPATIBILITY_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "accession_number",
    "subject_cik",
    "form",
    "acceptance_datetime",
    "availability_session",
    "filing_date",
    "filing_date_change",
    "official_complete_submission_url",
    "selected_document_identity",
    "primary_document_filename",
    "official_primary_document_url",
    "selected_text_start_byte",
    "selected_text_end_byte",
    "selected_text_length",
    "primary_document_body_kind",
    "source_record_sha256",
    "raw_primary_document_sha256",
    "normalized_text_sha256",
    "complete_response_sha256",
    "normalized_text_length",
    "complete_response_length",
)
PRIVATE_SOURCE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "source_url",
    "source_content_sha256",
    "raw_row_identity_sha256",
    "accession_number",
    "subject_cik",
    "form",
    "acceptance_datetime_source",
    "acceptance_datetime",
    "filing_date",
    "filing_date_change",
    "submissions_primary_document_raw",
    "submissions_filename",
    "sgml_filename",
    "sec_filename",
    "selected_document_identity",
    "complete_response_sha256",
)
NULLABLE_UNIVERSE_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "accession_number",
    "subject_cik",
    "form",
    "acceptance_datetime",
    "filing_date",
    "filing_date_change",
    "availability_session",
    "artifact_stage",
    "primary_document_filename",
    "selected_document_identity",
    "official_complete_submission_url",
    "official_primary_document_url",
    "source_record_sha256",
    "selected_text_sha256",
    "normalized_text_sha256",
    "complete_response_sha256",
    "compatibility_record_sha256",
)
NULLABLE_CONTENT_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "accession_number",
    "form",
    "availability_session",
    "primary_document_filename",
    "selected_document_identity",
    "official_complete_submission_url",
    "official_primary_document_url",
    "compatibility_record_sha256",
    "selected_text_sha256",
    "normalized_text_sha256",
    "complete_response_sha256",
    "selected_text_bytes",
    "normalized_text_bytes",
)
UNIVERSE_EVENT_PROOF_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "compatibility_manifest_sha256",
    "universe_sha256",
    "calendar_sessions_sha256",
    "current_record",
    "current_record_sha256",
    "current_content_record",
    "current_content_record_sha256",
    "current_filing_sha256",
    "prior_same_form_record",
    "prior_same_form_record_sha256",
    "prior_same_form_content_record",
    "prior_same_form_content_record_sha256",
    "prior_same_form_filing_sha256",
    "content_manifest_sha256",
    "prior_selection",
    "universe_event_proof_sha256",
)
BLINDED_MODEL_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "accession_number",
    "form",
    "availability_session",
    "preprocessed_event_sha256",
    "supplied_sentence_ids",
    "request_sha256",
    "request_bytes",
)
DIRECTION_SANITIZED_EVENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-direction-sanitized-event-v1"
)
DIRECTION_SANITIZER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-security-direction-filter-v1"
)
DIRECTION_SANITIZED_EVENT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "sanitizer_schema_version",
    "source_preprocessed_event_sha256",
    "source_sentence_count",
    "removed_current_sentence_count",
    "removed_prior_sentence_count",
    "retained_current_sentence_count",
    "retained_prior_sentence_count",
    "sentences",
    "sentences_sha256",
    "model_payload",
    "model_payload_sha256",
    "preprocessed_event_sha256",
)
_BLINDED_REQUEST_RECORD_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accession_number",
        "subject_cik",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "source_record_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
        "compatibility_record_sha256",
    }
)
_BLINDED_REQUEST_PROOF_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "compatibility_manifest_sha256",
        "universe_sha256",
        "calendar_sessions_sha256",
        "current_record_sha256",
        "current_content_record_sha256",
        "current_filing_sha256",
        "prior_same_form_record_sha256",
        "prior_same_form_content_record_sha256",
        "prior_same_form_filing_sha256",
        "content_manifest_sha256",
        "universe_event_proof_sha256",
    }
)
_BLINDED_REQUEST_MISSINGNESS_SUBJECTS: Final[tuple[str, ...]] = (
    "accession",
    "compatibility_record",
    "content_record",
    "document",
    "filename",
    "filing",
    "locator",
    "source_identity",
    "source_record",
    "url",
)
_BLINDED_TEXT_SOURCE_MISSINGNESS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"(?:accession(?: number)?|compatibility record|content record|"
    r"(?:official )?primary document(?: filename| url)?|document|filename|"
    r"filing|locator|source identity|source record|source url|url)"
    r"(?: field| value| entry)? "
    r"(?:is |was |were |remains |remained |does |has )?"
    r"(?:missing|missingness|absent|unavailable|omitted|null|none|"
    r"not present|not available|not supplied|not provided|nonexistent|"
    r"does not exist|has no(?: value| entry)?|present|available|exists|"
    r"supplied|provided|a value|an entry)"
    r"|"
    r"(?:missing|absent|unavailable|omitted|null|nonexistent|without|"
    r"available|present|supplied|provided) "
    r"(?:the )?"
    r"(?:accession(?: number)?|compatibility record|content record|"
    r"(?:official )?primary document(?: filename| url)?|document|filename|"
    r"filing|locator|source identity|source record|source url|url)"
    r"|"
    r"no (?:accession(?: number)?|compatibility record|content record|"
    r"(?:official )?primary document(?: filename| url)?|document|filename|"
    r"filing|locator|source identity|source record|source url|url)"
    r"(?: was| is)?(?: supplied| provided| available| present)?"
    r")"
)
_BLINDED_TEXT_ISSUER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9])(?:"
    r"a[^a-z0-9]*a[^a-z0-9]*p[^a-z0-9]*l|"
    r"a[^a-z0-9]*p[^a-z0-9]*p[^a-z0-9]*l[^a-z0-9]*e"
    r")(?![a-z0-9])"
)
_BLINDED_TEXT_SOURCE_VOCABULARY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"(?<![a-z0-9])f[^a-z0-9]*i[^a-z0-9]*l[^a-z0-9]*e[^a-z0-9]*"
    r"n[^a-z0-9]*a[^a-z0-9]*m[^a-z0-9]*e(?![a-z0-9])|"
    r"(?<![a-z0-9])u[^a-z0-9]*r[^a-z0-9]*l(?![a-z0-9])|"
    r"(?<![a-z0-9])accession(?![a-z0-9])|"
    r"(?<![a-z0-9])locator(?![a-z0-9])|"
    r"(?<![a-z0-9])source[^a-z0-9]+(?:identity|record|url|locator)(?![a-z0-9])|"
    r"(?<![a-z0-9])(?:official[^a-z0-9]+)?primary[^a-z0-9]+document(?![a-z0-9])"
    r")"
)
_BLINDED_TEXT_EXACT_DATE_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?<![a-z0-9])(?:[0-9]{1,4}[-/][0-9]{1,2}(?:[-/][0-9]{1,4})?)(?![a-z0-9])",
        r"(?<![a-z0-9])(?:19|20)[0-9]{2}(?![a-z0-9])",
        r"(?<![a-z0-9])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)(?![a-z0-9])",
        r"(?<![a-z0-9])may\s+[0-9]{1,2}(?![a-z0-9])",
    )
)
_BLINDED_TEXT_MARKET_CONTEXT_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?<![a-z0-9])(?:stock|share|security|market)\s+"
        r"(?:price|prices|return|returns|performance)(?![a-z0-9])",
        r"(?<![a-z0-9])(?:price|total|excess|market)\s+returns?(?![a-z0-9])",
        r"(?<![a-z0-9])total\s+shareholder\s+return(?![a-z0-9])",
        r"(?<![a-z0-9])benchmark\s+(?:comparison|performance|result|return)"
        r"(?![a-z0-9])",
        r"(?<![a-z0-9])(?:comparison|performance|result|return)\s+benchmark"
        r"(?![a-z0-9])",
        r"(?<![a-z0-9])buy[- ]and[- ]hold(?![a-z0-9])",
        r"(?<![a-z0-9])beat(?:s|ing)?\s+(?:the\s+)?market(?![a-z0-9])",
        r"(?<![a-z0-9])trading\s+"
        r"(?:action|position|return|signal|strategy)(?![a-z0-9])",
        r"(?<![a-z0-9])(?:long|short|cash)\s+"
        r"(?:action|exposure|position|signal)(?![a-z0-9])",
        r"(?<![a-z0-9])(?:buy|sell|hold|bullish|bearish)\s+"
        r"(?:action|label|position|rating|recommendation|signal)(?![a-z0-9])",
        r"(?<![a-z0-9])trading\s+(?:recommendation|advice)(?![a-z0-9])",
    )
)
_BLINDED_TEXT_MARKET_DIRECTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "advanced",
        "appreciated",
        "down",
        "dropped",
        "fell",
        "gained",
        "higher",
        "lost",
        "lower",
        "outperformed",
        "rallied",
        "rose",
        "surged",
        "underperformed",
        "up",
    }
)
_BLINDED_TEXT_SECURITY_DIRECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9])(?:"
    r"(?:stock|stocks|security|securities|share|shares|market)"
    r"(?:[ -]+(?:price|prices|value|values))?"
    r"[ -]+"
    r"(?:(?:are|closed|had|has|have|is|moved|moving|traded|was|were)[ -]+)?"
    r"(?:(?:considerably|dramatically|far|markedly|materially|moderately|modestly|much|notably|quite|sharply|significantly|slightly|somewhat|steeply|substantially|very)[ -]+){0,2}"
    r"(?:advanced|appreciated|down|dropped|fell|gained|higher|lost|lower|outperformed|rallied|rose|surged|underperformed|up)"
    r"|"
    r"(?:advanced|appreciated|down|dropped|fell|gained|higher|lost|lower|outperformed|rallied|rose|surged|underperformed|up)"
    r"[ -]+(?:stock|stocks|security|securities|share|shares|market)"
    r"(?:[ -]+(?:price|prices|value|values))?"
    r")(?![a-z0-9])"
)
_BLINDED_TEXT_MARKET_SHARE_SPAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9])market[ -]+shares?(?![a-z0-9])"
)
_BLINDED_TEXT_LABEL_VALUES: Final[frozenset[str]] = frozenset(
    {
        "bearish",
        "bullish",
        "buy",
        "cash",
        "high",
        "hold",
        "long",
        "low",
        "negative",
        "neutral",
        "positive",
        "sell",
        "short",
    }
)
_BLINDED_TEXT_KEYED_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9])(?:action|position|rating|decision|score|label|signal|"
    r"recommendation)\s*[:=]\s*(?:buy|sell|hold|long|short|cash|positive|"
    r"negative|neutral|high|low|bullish|bearish|risk\s+(?:on|off))"
    r"(?![a-z0-9])"
)
_BLINDED_TEXT_MODEL_ACTION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9])(?:the\s+)?model\s+(?:chose|selected|returned|predicted)\s+"
    r"(?:buy|sell|hold|long|short|cash|positive|negative|neutral|high|low|"
    r"bullish|bearish|risk\s+(?:on|off))(?![a-z0-9])"
)
_BLINDED_TEXT_BASE64_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{4,}={0,2}(?![A-Za-z0-9+/_=-])"
)
_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z"
)
_DATE_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_ACCEPTANCE_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{14}\Z")
_SAFE_PRIMARY_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z"
)
_TAGGED_SHA256_RE: Final[re.Pattern[str]] = re.compile(
    r"sha256:(?P<digest>[0-9a-f]{64})\Z"
)
PILOT_SORT_ORDER: Final[tuple[str, ...]] = (
    "canonical_request_byte_length_descending",
    "accession_ascending",
)

# Frozen local-model identity and request semantics.
MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
MODEL_CONFIG_DIGEST: Final[str] = (
    "c805f5b265d8e695c44f4065dfc368206cd8026447604925fef8db57ee32ee23"
)
MODEL_LAYER_DIGESTS: Final[tuple[str, ...]] = (
    "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
    "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
    "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
    "56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3",
)
MODEL_ACTIVE_FROM_BLOB_COUNT: Final[int] = 2
OLLAMA_VERSION: Final[str] = "0.32.0"
RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
)
RUNTIME_VERSION_RESPONSE_SHA256: Final[str] = (
    "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
)
RUNTIME_SHOW_SEMANTIC_SHA256: Final[str] = (
    "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
)
RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS: Final[tuple[str, ...]] = ("modified_at",)
RUNTIME_SHOW_RAW_SHA256_DIAGNOSTIC_ONLY: Final[str] = (
    "5f56fb0fb2214ddcb9fa21c66aa31e37297f553e8758aeda5958f0f287d70893"
)
RUNTIME_MODEL_INFO_SHA256: Final[str] = (
    "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
)
PROMPT_SHA256: Final[str] = (
    "9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7"
)
SCHEMA_SHA256: Final[str] = (
    "1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b"
)
MODEL_TEMPERATURE: Final[int] = 0
MODEL_SEED: Final[int] = 0
MODEL_CONTEXT_TOKENS: Final[int] = 6_144
MODEL_OUTPUT_TOKENS: Final[int] = 512
MODEL_INPUT_MAX_BYTES: Final[int] = 20_000
MODEL_REQUEST_MAX_BYTES: Final[int] = 128 * 1024
MODEL_INPUT_MAX_SENTENCES: Final[int] = 72
MODEL_INPUT_MAX_SENTENCE_CHARACTERS: Final[int] = 220
MODEL_RESPONSE_MAX_BYTES: Final[int] = 256 * 1024
OLLAMA_VERSION_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/version"
OLLAMA_SHOW_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/show"
OLLAMA_CHAT_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/chat"
RUNTIME_PROBE_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("GET", OLLAMA_VERSION_ENDPOINT),
    ("POST", OLLAMA_SHOW_ENDPOINT),
)
NORMAL_IDENTITY_REQUEST_COUNT: Final[int] = 4
PAUSED_RESUMED_IDENTITY_REQUEST_COUNT: Final[int] = 8

# Frozen Yahoo batch.
YAHOO_ENDPOINT: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)
YAHOO_SYMBOL_ORDER: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
)
YAHOO_PROVIDER_SYMBOLS: Final[tuple[tuple[str, str], ...]] = (
    ("AAPL", "AAPL"),
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("IWM", "IWM"),
    ("VIX", "^VIX"),
    ("TNX", "^TNX"),
)
YAHOO_QUERY_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("period1", "883612800"),
    ("period2", "1546300800"),
    ("interval", "1d"),
    ("includePrePost", "false"),
    ("includeAdjustedClose", "true"),
    ("events", "div,splits"),
)
YAHOO_URLS: Final[tuple[str, ...]] = (
    f"{YAHOO_ENDPOINT}/AAPL?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/SPY?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/QQQ?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/IWM?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/%5EVIX?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/%5ETNX?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
)
YAHOO_REQUEST_COUNT: Final[int] = 6
YAHOO_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
YAHOO_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
YAHOO_MAX_TOTAL_RESPONSE_BYTES: Final[int] = 128 * 1024 * 1024
YAHOO_MAX_BATCH_SECONDS: Final[int] = 210
YAHOO_USER_AGENT: Final[str] = (
    "LLM-memory-trading-agent/1.0 market-evidence (no-auth; one-shot)"
)

# Fixed pilot rule.
PILOT_PROJECTED_THRESHOLD_NS: Final[int] = 43_200_000_000_000
PILOT_PROJECTION_MULTIPLIER: Final[int] = 70

# One-shot and publication paths.
DEVELOPMENT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-development-001"
)
PRIVATE_NAMESPACE: Final[str] = "data/aapl_sec_gemma_lean_science_v3_19"
PRIVATE_PREFLIGHT_NAMESPACE: Final[str] = f"{PRIVATE_NAMESPACE}/preflight"
PRIVATE_DEVELOPMENT_NAMESPACE: Final[str] = f"{PRIVATE_NAMESPACE}/development"
PUBLIC_EVIDENCE_ROOT: Final[str] = "e/aapl_sec_gemma_lean_science_v3_19"
PREFLIGHT_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_PREFLIGHT.json"
)
PAUSE_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_PAUSE.json"
)
RESULT_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_RESULT.json"
)
COMPARISON_PATH: Final[str] = "e/APPROACH_COMPARISON.md"
CONTINUATION_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_19_continuation.md"
)

IMPLEMENTATION_PRODUCTION_PATHS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_gemma_lean_science_v319_contract.py",
    "agent_benchmark/sec_gemma_lean_science_v319_bridge.py",
    "agent_benchmark/sec_gemma_lean_science_v319_journal.py",
    "agent_benchmark/sec_gemma_lean_science_v319_store.py",
    "agent_benchmark/sec_gemma_lean_science_v319_preflight.py",
    "agent_benchmark/sec_gemma_lean_science_v319_runner.py",
)
IMPLEMENTATION_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_sec_gemma_lean_science_v319_contract.py",
    "tests/test_sec_gemma_lean_science_v319_bridge.py",
    "tests/test_sec_gemma_lean_science_v319_journal.py",
    "tests/test_sec_gemma_lean_science_v319_store.py",
    "tests/test_sec_gemma_lean_science_v319_preflight.py",
    "tests/test_sec_gemma_lean_science_v319_runner.py",
)
IMPLEMENTATION_ALLOWED_PATHS: Final[tuple[str, ...]] = (
    *IMPLEMENTATION_PRODUCTION_PATHS,
    *IMPLEMENTATION_TEST_PATHS,
)

# Frozen latest-approach/shared-dependency qualification authority.
LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-local-production-closure-v1"
)
LOCAL_PRODUCTION_CLOSURE_DERIVATION: Final[str] = (
    "python_ast_local_import_closure_v1"
)
LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_filing_content.py",
    "agent_benchmark/sec_filing_gemma_contract.py",
    "agent_benchmark/sec_gemma_lean_v38_journal.py",
    "agent_benchmark/sec_gemma_lean_v38_source.py",
    "agent_benchmark/sec_gemma_lean_v38_transport.py",
    "agent_benchmark/sec_session_calendar.py",
    "agent_benchmark/sec_filing_gemma_ollama.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_acquisition.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_features.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_production.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_runner.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_runtime.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_contract.py",
)
LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_filing_gemma_learner.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_production.py",
)
LOCAL_PRODUCTION_CLOSURE_PATHS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_audit_transport.py",
    "agent_benchmark/sec_filing_content.py",
    "agent_benchmark/sec_filing_gemma_contract.py",
    "agent_benchmark/sec_filing_gemma_corpus.py",
    "agent_benchmark/sec_filing_gemma_extractor_prompt.py",
    "agent_benchmark/sec_filing_gemma_extractor_schema.py",
    "agent_benchmark/sec_filing_gemma_learner.py",
    "agent_benchmark/sec_filing_gemma_market_acquirer.py",
    "agent_benchmark/sec_filing_gemma_market_evidence.py",
    "agent_benchmark/sec_filing_gemma_market_source_bytes.py",
    "agent_benchmark/sec_filing_gemma_ollama.py",
    "agent_benchmark/sec_filing_gemma_preprocessor.py",
    "agent_benchmark/sec_gemma_lean_v38_journal.py",
    "agent_benchmark/sec_gemma_lean_v38_source.py",
    "agent_benchmark/sec_gemma_lean_v38_transport.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_acquisition.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_attempt.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_baseline.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_contract.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_features.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_learner.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_ledger.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_market_verifier.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_metrics.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_no_leverage.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_policy.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_production.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_publisher.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_registry.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_replay.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_runner.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_runtime.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_source_verifier.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_store.py",
    "agent_benchmark/sec_gemma_online_risk_overlay_vault.py",
    "agent_benchmark/sec_point_in_time.py",
    "agent_benchmark/sec_session_calendar.py",
)
LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES: Final[int] = 8_756
LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256: Final[str] = (
    "863a293ee604e855d480230068f9e53b609aed36e1af2d2d316b8b87bdeedf2a"
)
LOCAL_PRODUCTION_CLOSURE_PATHS_BYTES: Final[int] = 2_023
LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256: Final[str] = (
    "0a1e7374b35677596418cfa82575bd6614df631eba34f1978ce9c9017a74aae9"
)
LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH: Final[str] = "agent_benchmark/__init__.py"
LOCAL_IMPORT_ALLOWED_PATHS: Final[tuple[str, ...]] = (
    LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    *IMPLEMENTATION_PRODUCTION_PATHS,
    *LOCAL_PRODUCTION_CLOSURE_PATHS,
)

QUALIFICATION_SUITE_ID: Final[str] = (
    "latest_approach_actual_dependencies_pinned_parent_authority"
)
QUALIFICATION_PHASE_V319: Final[str] = "v319"
QUALIFICATION_PHASE_DEPENDENCIES: Final[str] = "dependencies"
QUALIFICATION_PHASE_REQUESTS_IDENTITY: Final[str] = "requests_identity"
QUALIFICATION_PHASES: Final[tuple[str, ...]] = (
    QUALIFICATION_PHASE_V319,
    QUALIFICATION_PHASE_DEPENDENCIES,
    QUALIFICATION_PHASE_REQUESTS_IDENTITY,
)
QUALIFICATION_MODES: Final[tuple[str, ...]] = ("collection", "execution")
SCIENTIFIC_ENVIRONMENT_NAMES: Final[tuple[str, ...]] = (
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "PATH",
    "PYTHONNOUSERSITE",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "TZ",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
SCIENTIFIC_ENVIRONMENT_FIXED_VALUES: Final[tuple[tuple[str, str], ...]] = (
    ("PYTHONNOUSERSITE", "1"),
    ("PYTHONHASHSEED", "0"),
    ("PYTHONPATH", ""),
    ("PYTHONUTF8", "1"),
    ("PYTHONIOENCODING", "utf-8"),
    ("TZ", "America/New_York"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
)
QUALIFICATION_ENVIRONMENT: Final[tuple[tuple[str, str], ...]] = (
    *SCIENTIFIC_ENVIRONMENT_FIXED_VALUES,
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
)
QUALIFICATION_ENVIRONMENT_NAMES: Final[tuple[str, ...]] = (
    *SCIENTIFIC_ENVIRONMENT_NAMES,
    "PYTHONDONTWRITEBYTECODE",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)
QUALIFICATION_CONTROLLED_PYTEST_ENVIRONMENT: Final[
    tuple[tuple[str, str], ...]
] = (
    ("USERPROFILE", "canonical_TEMP"),
    ("LOCALAPPDATA", "canonical_TEMP"),
    ("APPDATA", "canonical_TEMP"),
    ("HOME", "canonical_TEMP"),
    ("COMSPEC", "canonical_SYSTEMROOT/System32/cmd.exe"),
    ("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
)
QUALIFICATION_PYTEST_ENVIRONMENT_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            *QUALIFICATION_ENVIRONMENT_NAMES,
            *(
                name
                for name, _authority
                in QUALIFICATION_CONTROLLED_PYTEST_ENVIRONMENT
            ),
        ),
        key=lambda name: name.encode("utf-8"),
    )
)
QUALIFICATION_PYTHON_FLAGS: Final[tuple[str, ...]] = ("-s", "-S", "-B")
QUALIFICATION_BOOTSTRAP_LITERAL: Final[str] = """import os,sys

def fail(code):
    raise SystemExit(code)

flag_names=('debug','inspect','interactive','optimize','dont_write_bytecode','no_user_site','no_site','ignore_environment','verbose','bytes_warning','quiet','hash_randomization','isolated','dev_mode','utf8_mode','warn_default_encoding','safe_path','int_max_str_digits')
flag_values=(0,0,0,0,1,1,1,0,0,0,0,0,0,False,1,0,False,4300)
if tuple(getattr(sys.flags,n) for n in flag_names)!=flag_values:
    fail('bootstrap_flags_invalid')
env_values={'PYTHONNOUSERSITE':'1','PYTHONHASHSEED':'0','PYTHONPATH':'','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','TZ':'America/New_York','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
env_names=tuple(sorted(('SYSTEMROOT','WINDIR','TEMP','TMP','PATH',*env_values)))
if tuple(sorted(os.environ))!=env_names or any(os.environ.get(k)!=v for k,v in env_values.items()) or any(not os.environ.get(k) for k in ('SYSTEMROOT','WINDIR','TEMP','TMP','PATH')):
    fail('bootstrap_environment_invalid')
if any(n in sys.modules for n in ('site','sitecustomize','usercustomize')):
    fail('bootstrap_customization_loaded')
key=lambda p:os.path.normcase(os.path.normpath(p))
exe=os.path.realpath(sys.executable,strict=True)
parent=os.path.dirname(exe)
zip_path=os.path.join(parent,'python312.zip')
initial=('',zip_path,os.path.join(parent,'DLLs'),os.path.join(parent,'Lib'),parent)
if len(sys.path)!=5 or sys.path[0]!='' or any(key(a)!=key(b) for a,b in zip(sys.path,initial)):
    fail('bootstrap_initial_sys_path_invalid')
if os.path.lexists(zip_path) or any(not os.path.isdir(p) for p in initial[2:]):
    fail('bootstrap_initial_sys_path_state_invalid')
if len(sys.argv)<7 or sys.argv[1]!='pytest' or sys.argv[2]!='--repo-root' or sys.argv[4]!='--':
    fail('bootstrap_argv_invalid')
root_arg=sys.argv[3]
pytest_args=sys.argv[5:]
if not os.path.isabs(root_arg) or not pytest_args:
    fail('bootstrap_argv_invalid')

def resolve_dir(path):
    resolved=os.path.realpath(path,strict=True)
    if not os.path.isdir(resolved):
        fail('bootstrap_path_invalid')
    return resolved

root=resolve_dir(root_arg)
git_dir=os.path.join(root,'.git')
git_stat=os.lstat(git_dir)
if key(root_arg)!=key(root) or not os.path.samefile(root,os.getcwd()) or not os.path.isdir(git_dir) or os.path.islink(git_dir) or getattr(git_stat,'st_file_attributes',0)&1024:
    fail('bootstrap_repo_root_invalid')
final=[]
seen=set()
def add(path):
    resolved=resolve_dir(path)
    token=key(resolved)
    if token not in seen:
        seen.add(token)
        final.append(resolved)
add(root)
for path in initial[2:]:
    add(path)
sys.path[:]=final
import hashlib,stat,sysconfig
if any(n in sys.modules for n in ('site','sitecustomize','usercustomize')):
    fail('bootstrap_customization_loaded')
paths=sysconfig.get_paths()
for name in ('purelib','platlib'):
    if name not in paths:
        fail('bootstrap_install_path_missing')
    add(paths[name])
sys.path[:]=final
import pytest
purelib=resolve_dir(paths['purelib'])
pytest_path=os.path.abspath(pytest.__file__)
expected_pytest_path=os.path.join(purelib,'pytest','__init__.py')
if pytest.__version__!='9.0.3' or key(pytest_path)!=key(expected_pytest_path) or key(os.path.realpath(pytest_path,strict=True))!=key(expected_pytest_path):
    fail('bootstrap_pytest_identity_invalid')
pytest_stat=os.lstat(pytest_path)
if not stat.S_ISREG(pytest_stat.st_mode) or os.path.islink(pytest_path) or getattr(pytest_stat,'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT:
    fail('bootstrap_pytest_identity_invalid')
with open(pytest_path,'rb') as handle:
    pytest_bytes=handle.read()
if len(pytest_bytes)!=5582 or hashlib.sha256(pytest_bytes).hexdigest()!='7be7a1e2218dc59a19d1ad131e4abe21172a295087efc72898938248782e8766' or any(n in sys.modules for n in ('site','sitecustomize','usercustomize')):
    fail('bootstrap_pytest_identity_invalid')
qualification_temp=os.path.realpath(os.environ['TEMP'],strict=True)
qualification_system_root=os.path.realpath(os.environ['SYSTEMROOT'],strict=True)
qualification_windir=os.path.realpath(os.environ['WINDIR'],strict=True)
qualification_tmp=os.path.realpath(os.environ['TMP'],strict=True)
qualification_comspec=os.path.join(qualification_system_root,'System32','cmd.exe')
if key(qualification_temp)!=key(os.environ['TEMP']) or key(qualification_tmp)!=key(qualification_temp) or key(qualification_system_root)!=key(os.environ['SYSTEMROOT']) or key(qualification_windir)!=key(qualification_system_root) or not os.path.isdir(qualification_temp) or not os.path.isdir(qualification_system_root) or key(os.path.realpath(qualification_comspec,strict=True))!=key(qualification_comspec) or not os.path.isfile(qualification_comspec):
    fail('bootstrap_windows_recovery_environment_invalid')
controlled_environment={'USERPROFILE':qualification_temp,'LOCALAPPDATA':qualification_temp,'APPDATA':qualification_temp,'HOME':qualification_temp,'COMSPEC':qualification_comspec,'PATHEXT':'.COM;.EXE;.BAT;.CMD'}
if any(name in os.environ for name in controlled_environment):
    fail('bootstrap_windows_recovery_environment_invalid')
os.environ.update(controlled_environment)
if tuple(sorted(os.environ))!=tuple(sorted((*env_names,*controlled_environment))) or any(os.environ.get(name)!=value for name,value in controlled_environment.items()):
    fail('bootstrap_windows_recovery_environment_invalid')
raise SystemExit(pytest.main(pytest_args))"""
QUALIFICATION_BOOTSTRAP_BYTES: Final[int] = 5_482
QUALIFICATION_BOOTSTRAP_SHA256: Final[str] = (
    "59aa7f29b7200e5a915b15a840da050e7803c825bf40ad42720dc0a4092e4407"
)
QUALIFICATION_CANONICAL_ROOT_TOKEN: Final[str] = "{canonical_absolute_root}"
QUALIFICATION_DISPATCH_PREFIX: Final[tuple[str, ...]] = (
    "pytest",
    "--repo-root",
    QUALIFICATION_CANONICAL_ROOT_TOKEN,
    "--",
)
QUALIFICATION_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "-q",
    "-p",
    "no:cacheprovider",
)
QUALIFICATION_JUNIT_RELATIVE_TEMPLATE: Final[str] = (
    "qualification/{phase}/{mode}/junit.xml.partial"
)
QUALIFICATION_LATEST_TEST_PATHS: Final[tuple[str, ...]] = (
    *IMPLEMENTATION_TEST_PATHS,
)
QUALIFICATION_LATEST_MIN_NODE_COUNT: Final[int] = 333
QUALIFICATION_SHARED_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_sec_filing_content.py",
    "tests/test_sec_filing_gemma_contract.py",
    "tests/test_sec_point_in_time.py",
    "tests/test_sec_gemma_lean_v38_journal.py",
    "tests/test_sec_gemma_lean_v38_source.py",
    "tests/test_sec_gemma_lean_v38_transport.py",
    "tests/test_sec_session_calendar.py",
    "tests/test_sec_filing_gemma_ollama.py",
    "tests/test_sec_gemma_online_risk_overlay_acquisition.py",
    "tests/test_sec_gemma_online_risk_overlay_features.py",
    "tests/test_sec_gemma_online_risk_overlay_production.py",
    "tests/test_sec_gemma_online_risk_overlay_runner.py",
    "tests/test_sec_gemma_online_risk_overlay_runtime.py",
    "tests/test_sec_gemma_online_risk_overlay_contract.py",
)
QUALIFICATION_SHARED_NODE_COUNT: Final[int] = 655
QUALIFICATION_SHARED_UNIQUE_NODE_COUNT: Final[int] = 655
QUALIFICATION_SHARED_DUPLICATE_NODE_COUNT: Final[int] = 0
QUALIFICATION_SHARED_MULTIPLICITY_POLICY: Final[str] = (
    "python_unicode_case_sensitive_collections_counter_v1"
)
QUALIFICATION_SHARED_NODE_LIST_SHA256: Final[str] = (
    "1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c"
)
QUALIFICATION_SENTINEL_NODE_ID: Final[str] = (
    "tests/test_sec_gemma_lean_runner.py::"
    "test_runtime_modules_share_exact_verified_requests_identity"
)
QUALIFICATION_SENTINEL_NODE_COUNT: Final[int] = 1
QUALIFICATION_SENTINEL_NODE_LIST_SHA256: Final[str] = (
    "465bbb7fb1bd0633502006db2b84f6adab6ca7542d8c4e3d75b13d6cb7e73229"
)
QUALIFICATION_COLLECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-qualification-collection-v1"
)
QUALIFICATION_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-qualification-intent-v1"
)
QUALIFICATION_RESULT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-qualification-result-v1"
)
QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-qualification-private-aggregate-v1"
)
QUALIFICATION_PUBLIC_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-19-qualification-public-receipt-v1"
)
QUALIFICATION_PYTHON_VERSION: Final[str] = (
    "3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) "
    "[MSC v.1937 64 bit (AMD64)]"
)
QUALIFICATION_PYTHON_CACHE_TAG: Final[str] = "cpython-312"
QUALIFICATION_OS_NAME: Final[str] = "nt"
QUALIFICATION_SYS_PLATFORM: Final[str] = "win32"
QUALIFICATION_EXECUTABLE_BASENAME: Final[str] = "python.exe"
QUALIFICATION_EXECUTABLE_BYTES: Final[int] = 103_192
QUALIFICATION_EXECUTABLE_SHA256: Final[str] = (
    "624bbc0586d8855633b875e911883bbef8a0e8b8711e11126df480dd86f54181"
)
QUALIFICATION_PYTEST_VERSION: Final[str] = "9.0.3"
QUALIFICATION_PYTEST_INIT_BYTES: Final[int] = 5_582
QUALIFICATION_PYTEST_INIT_SHA256: Final[str] = (
    "7be7a1e2218dc59a19d1ad131e4abe21172a295087efc72898938248782e8766"
)
QUALIFICATION_TIMEOUT_SECONDS: Final[int] = 25 * 60
QUALIFICATION_DURABILITY_MODE: Final[str] = (
    "windows_file_fsync_rename_noreplace_marker_last_v1"
)


EFFECT_COUNT_KEYS: Final[tuple[str, ...]] = (
    "sec_requests",
    "experiment_family_sec_requests",
    "yahoo_requests",
    "ollama_identity_http_requests",
    "ollama_chat_generations",
    "retries",
    "repairs",
    "pulls",
    "fallbacks",
    "paid_calls",
    "confirmation_final_data_opens",
    "broker_effects",
    "real_money_effects",
)


def _effect_budget(*, identity_requests: int, generations: int) -> dict[str, int]:
    return {
        "sec_requests": 0,
        "experiment_family_sec_requests": EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
        "yahoo_requests": YAHOO_REQUEST_COUNT,
        "ollama_identity_http_requests": identity_requests,
        "ollama_chat_generations": generations,
        "retries": 0,
        "repairs": 0,
        "pulls": 0,
        "fallbacks": 0,
        "paid_calls": 0,
        "confirmation_final_data_opens": 0,
        "broker_effects": 0,
        "real_money_effects": 0,
    }


NORMAL_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_DOCUMENT_COUNT,
)
PAUSED_RESUMED_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=PAUSED_RESUMED_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_DOCUMENT_COUNT,
)
PILOT_PAUSE_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_PILOT_COUNT,
)
ZERO_EFFECT_BUDGET: Final[dict[str, int]] = {
    key: (EXPERIMENT_FAMILY_SEC_REQUEST_COUNT if key == "experiment_family_sec_requests" else 0)
    for key in EFFECT_COUNT_KEYS
}

HIGH_LEVEL_EFFECT_ORDER: Final[tuple[str, ...]] = (
    "authenticate_preregistration_and_implementation",
    "authenticate_v38_source_read_only",
    "rebuild_projection_requests_and_pilot_commitments",
    "consume_one_shot_attempt",
    "seal_six_yahoo_responses_without_opening_values",
    "model_batch_pre_probe",
    "seal_five_pilots",
    "apply_timing_only_pause_rule",
    "seal_remaining_seventy_if_not_paused",
    "model_batch_post_probe",
    "open_and_validate_complete_market_and_semantic_batches",
    "run_frozen_deterministic_science",
    "independent_replay_and_no_leverage_proofs",
    "seal_redacted_terminal_evidence",
)


def _reject(code: str) -> None:
    raise ContractViolation(code)


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 128:
        _reject("v319_contract_json_depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _reject("v319_contract_json_nonfinite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _reject("v319_contract_json_key")
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    _reject("v319_contract_json_type")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the preregistered UTF-8 canonical JSON representation."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractViolation("v319_contract_json_encoding") from exc


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return a lowercase unprefixed SHA-256 for a bytes-like value."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        _reject("v319_contract_hash_input")
    return hashlib.sha256(bytes(value)).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def add_self_sha256(
    value: Mapping[str, Any],
    *,
    field: str = "manifest_sha256",
) -> dict[str, Any]:
    """Detach a mapping and bind its exact unsigned canonical JSON."""

    if not isinstance(value, Mapping) or not isinstance(field, str) or not field:
        _reject("v319_contract_self_hash_shape")
    if field in value:
        _reject("v319_contract_self_hash_present")
    detached = copy.deepcopy(dict(value))
    detached[field] = canonical_sha256(detached)
    return detached


def validate_self_sha256(
    value: Any,
    *,
    field: str = "manifest_sha256",
) -> dict[str, Any]:
    """Validate a strict lowercase SHA-256 self-hash and detach the mapping."""

    if not isinstance(value, Mapping) or not isinstance(field, str) or not field:
        _reject("v319_contract_self_hash_shape")
    detached = copy.deepcopy(dict(value))
    observed = detached.pop(field, None)
    if not _is_hex(observed, 64) or observed != canonical_sha256(detached):
        _reject("v319_contract_self_hash_mismatch")
    return copy.deepcopy(dict(value))


def strip_exact_sha256_tag(value: Any) -> str:
    """Accept exactly ``sha256:`` plus one lowercase 64-hex digest."""

    match = _TAGGED_SHA256_RE.fullmatch(value) if type(value) is str else None
    if match is None:
        _reject("v319_contract_tagged_sha256")
    return match.group("digest")


def _validate_nullable_locator_fields(value: Mapping[str, Any], *, code: str) -> None:
    try:
        accession = value["accession_number"]
        filename = value["primary_document_filename"]
        identity = value["selected_document_identity"]
        primary_url = value["official_primary_document_url"]
    except Exception:
        _reject(code)
    if type(accession) is not str or _ACCESSION_RE.fullmatch(accession) is None:
        _reject(code)
    if filename is None:
        if primary_url is not None or identity != LEGACY_MISSING_DOCUMENT_IDENTITY:
            _reject(code)
        return
    if (
        type(filename) is not str
        or _SAFE_PRIMARY_FILENAME_RE.fullmatch(filename) is None
        or identity != filename
    ):
        _reject(code)
    expected = (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession.replace('-', '')}/"
        f"{quote(filename, safe='-._~', encoding='utf-8', errors='strict')}"
    )
    if primary_url != expected:
        _reject(code)


def validate_compatibility_record(value: Any) -> dict[str, Any]:
    """Validate one exact 22-field nullable V3.8 compatibility record."""

    item = _strict_mapping(
        value,
        frozenset(COMPATIBILITY_RECORD_FIELDS),
        code="v319_contract_compatibility_record_shape",
    )
    _validate_nullable_locator_fields(
        item, code="v319_contract_compatibility_record_locator"
    )
    accession = item["accession_number"]
    expected_complete_url = (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession}.txt"
    )
    tagged_fields = (
        "raw_primary_document_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
    )
    if (
        item["schema_version"] != NULLABLE_SOURCE_RECORD_SCHEMA_VERSION
        or item["subject_cik"] != "0000320193"
        or item["form"] not in {"10-K", "10-Q"}
        or type(item["acceptance_datetime"]) is not str
        or _ACCEPTANCE_RE.fullmatch(item["acceptance_datetime"]) is None
        or type(item["availability_session"]) is not str
        or _DATE_RE.fullmatch(item["availability_session"]) is None
        or type(item["filing_date"]) is not str
        or _DATE_RE.fullmatch(item["filing_date"]) is None
        or (
            item["filing_date_change"] is not None
            and (
                type(item["filing_date_change"]) is not str
                or _DATE_RE.fullmatch(item["filing_date_change"]) is None
            )
        )
        or item["official_complete_submission_url"] != expected_complete_url
        or item["primary_document_body_kind"] != PRIMARY_DOCUMENT_BODY_KIND
        or not _is_hex(item["source_record_sha256"], 64)
        or any(
            type(item[field]) is not str
            or _TAGGED_SHA256_RE.fullmatch(item[field]) is None
            for field in tagged_fields
        )
        or type(item["selected_text_start_byte"]) is not int
        or type(item["selected_text_end_byte"]) is not int
        or type(item["selected_text_length"]) is not int
        or item["selected_text_start_byte"] < 0
        or item["selected_text_length"] <= 0
        or item["selected_text_end_byte"] - item["selected_text_start_byte"]
        != item["selected_text_length"]
        or type(item["normalized_text_length"]) is not int
        or item["normalized_text_length"] <= 0
        or type(item["complete_response_length"]) is not int
        or item["complete_response_length"] <= 0
        or item["selected_text_end_byte"] > item["complete_response_length"]
    ):
        _reject("v319_contract_compatibility_record_identity")
    return item


def validate_compatibility_manifest(value: Any) -> dict[str, Any]:
    """Validate the exact private 75-row compatibility-manifest authority."""

    fields = frozenset(
        {
            "schema_version",
            "source_authority_pins_sha256",
            "science_contract_projection_sha256",
            "legacy_source_projection_sha256",
            "document_count",
            "filename_present_count",
            "filename_missing_count",
            "ordered_compatibility_record_sha256s",
            "nullable_filename_bitmap_sha256",
            "source_order_sha256",
            "event_order_sha256",
            "prior_links_sha256",
            "compatibility_manifest_sha256",
        }
    )
    item = _strict_mapping(
        value, fields, code="v319_contract_compatibility_manifest_shape"
    )
    validate_self_sha256(item, field="compatibility_manifest_sha256")
    hashes = item["ordered_compatibility_record_sha256s"]
    if (
        item["schema_version"] != COMPATIBILITY_MANIFEST_SCHEMA_VERSION
        or item["source_authority_pins_sha256"] != SOURCE_AUTHORITY_PINS_SHA256
        or item["science_contract_projection_sha256"] != SCIENCE_PROJECTION_SHA256
        or not _is_hex(item["legacy_source_projection_sha256"], 64)
        or item["document_count"] != DEVELOPMENT_DOCUMENT_COUNT
        or item["filename_present_count"] != DEVELOPMENT_FILENAME_PRESENT_COUNT
        or item["filename_missing_count"] != DEVELOPMENT_FILENAME_MISSING_COUNT
        or type(hashes) is not list
        or len(hashes) != DEVELOPMENT_DOCUMENT_COUNT
        or len(set(hashes)) != DEVELOPMENT_DOCUMENT_COUNT
        or any(not _is_hex(child, 64) for child in hashes)
        or any(
            not _is_hex(item[field], 64)
            for field in (
                "nullable_filename_bitmap_sha256",
                "source_order_sha256",
                "event_order_sha256",
                "prior_links_sha256",
            )
        )
    ):
        _reject("v319_contract_compatibility_manifest_identity")
    return item


def validate_nullable_universe_record(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        frozenset(NULLABLE_UNIVERSE_RECORD_FIELDS),
        code="v319_contract_nullable_universe_record_shape",
    )
    _validate_nullable_locator_fields(
        item, code="v319_contract_nullable_universe_record_locator"
    )
    accession = item["accession_number"]
    if (
        item["subject_cik"] != "0000320193"
        or item["form"] not in {"10-K", "10-Q"}
        or type(item["acceptance_datetime"]) is not str
        or _ACCEPTANCE_RE.fullmatch(item["acceptance_datetime"]) is None
        or type(item["filing_date"]) is not str
        or _DATE_RE.fullmatch(item["filing_date"]) is None
        or (
            item["filing_date_change"] is not None
            and (
                type(item["filing_date_change"]) is not str
                or _DATE_RE.fullmatch(item["filing_date_change"]) is None
            )
        )
        or type(item["availability_session"]) is not str
        or _DATE_RE.fullmatch(item["availability_session"]) is None
        or item["artifact_stage"] != "development"
        or item["official_complete_submission_url"]
        != "https://www.sec.gov/Archives/edgar/data/320193/" + accession + ".txt"
        or any(
            not _is_hex(item[field], 64)
            for field in (
                "source_record_sha256",
                "selected_text_sha256",
                "normalized_text_sha256",
                "complete_response_sha256",
                "compatibility_record_sha256",
            )
        )
    ):
        _reject("v319_contract_nullable_universe_record_identity")
    return item


def validate_nullable_universe_manifest(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "source_authority_pins_sha256",
            "science_contract_projection_sha256",
            "legacy_source_projection_sha256",
            "compatibility_manifest_sha256",
            "calendar_sessions_sha256",
            "document_count",
            "exact_acceptance_timestamp_count",
            "stage_counts",
            "records",
            "universe_semantic_sha256",
            "universe_sha256",
        }
    )
    item = _strict_mapping(
        value, fields, code="v319_contract_nullable_universe_shape"
    )
    validate_self_sha256(item, field="universe_sha256")
    raw_records = item["records"]
    if type(raw_records) is not list:
        _reject("v319_contract_nullable_universe_records")
    records = [validate_nullable_universe_record(child) for child in raw_records]
    accessions = [child["accession_number"] for child in records]
    ordered = sorted(
        records,
        key=lambda child: (child["availability_session"], child["accession_number"]),
    )
    semantic = {
        "schema_version": NULLABLE_UNIVERSE_SEMANTIC_SCHEMA_VERSION,
        "calendar_sessions_sha256": CALENDAR_SESSIONS_SHA256,
        "records": [
            {
                "accession_number": child["accession_number"],
                "form": child["form"],
                "availability_session": child["availability_session"],
                "artifact_stage": child["artifact_stage"],
            }
            for child in records
        ],
    }
    if (
        item["schema_version"] != NULLABLE_UNIVERSE_SCHEMA_VERSION
        or item["source_authority_pins_sha256"] != SOURCE_AUTHORITY_PINS_SHA256
        or item["science_contract_projection_sha256"] != SCIENCE_PROJECTION_SHA256
        or not _is_hex(item["legacy_source_projection_sha256"], 64)
        or not _is_hex(item["compatibility_manifest_sha256"], 64)
        or item["calendar_sessions_sha256"] != CALENDAR_SESSIONS_SHA256
        or item["document_count"] != DEVELOPMENT_DOCUMENT_COUNT
        or item["exact_acceptance_timestamp_count"] != DEVELOPMENT_DOCUMENT_COUNT
        or item["stage_counts"] != {"development": DEVELOPMENT_DOCUMENT_COUNT}
        or len(records) != DEVELOPMENT_DOCUMENT_COUNT
        or len(set(accessions)) != DEVELOPMENT_DOCUMENT_COUNT
        or records != ordered
        or item["universe_semantic_sha256"] != canonical_sha256(semantic)
    ):
        _reject("v319_contract_nullable_universe_identity")
    return item


def validate_nullable_content_record(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        frozenset(NULLABLE_CONTENT_RECORD_FIELDS),
        code="v319_contract_nullable_content_record_shape",
    )
    _validate_nullable_locator_fields(
        item, code="v319_contract_nullable_content_record_locator"
    )
    accession = item["accession_number"]
    if (
        item["form"] not in {"10-K", "10-Q"}
        or type(item["availability_session"]) is not str
        or _DATE_RE.fullmatch(item["availability_session"]) is None
        or item["official_complete_submission_url"]
        != "https://www.sec.gov/Archives/edgar/data/320193/" + accession + ".txt"
        or any(
            not _is_hex(item[field], 64)
            for field in (
                "compatibility_record_sha256",
                "selected_text_sha256",
                "normalized_text_sha256",
                "complete_response_sha256",
            )
        )
        or type(item["selected_text_bytes"]) is not int
        or item["selected_text_bytes"] <= 0
        or type(item["normalized_text_bytes"]) is not int
        or item["normalized_text_bytes"] <= 0
    ):
        _reject("v319_contract_nullable_content_record_identity")
    return item


def validate_nullable_content_manifest(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "artifact_stage",
            "universe_sha256",
            "compatibility_manifest_sha256",
            "document_count",
            "documents",
            "content_manifest_sha256",
        }
    )
    item = _strict_mapping(
        value, fields, code="v319_contract_nullable_content_shape"
    )
    validate_self_sha256(item, field="content_manifest_sha256")
    raw_documents = item["documents"]
    if type(raw_documents) is not list:
        _reject("v319_contract_nullable_content_documents")
    documents = [validate_nullable_content_record(child) for child in raw_documents]
    accessions = [child["accession_number"] for child in documents]
    if (
        item["schema_version"] != NULLABLE_CONTENT_SCHEMA_VERSION
        or item["artifact_stage"] != "development"
        or not _is_hex(item["universe_sha256"], 64)
        or not _is_hex(item["compatibility_manifest_sha256"], 64)
        or item["document_count"] != DEVELOPMENT_DOCUMENT_COUNT
        or len(documents) != DEVELOPMENT_DOCUMENT_COUNT
        or len(set(accessions)) != DEVELOPMENT_DOCUMENT_COUNT
        or documents
        != sorted(
            documents,
            key=lambda child: (
                child["availability_session"],
                child["accession_number"],
            ),
        )
    ):
        _reject("v319_contract_nullable_content_identity")
    return item


def _compatibility_record_for_successor(
    record: Mapping[str, Any],
    compatibility_by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    digest = record["compatibility_record_sha256"]
    compatibility = compatibility_by_hash.get(digest)
    if compatibility is None:
        _reject("v319_contract_proof_compatibility_missing")
    item = validate_compatibility_record(compatibility)
    expected = {
        "accession_number": item["accession_number"],
        "subject_cik": item["subject_cik"],
        "form": item["form"],
        "acceptance_datetime": item["acceptance_datetime"],
        "filing_date": item["filing_date"],
        "filing_date_change": item["filing_date_change"],
        "availability_session": item["availability_session"],
        "primary_document_filename": item["primary_document_filename"],
        "selected_document_identity": item["selected_document_identity"],
        "official_complete_submission_url": item["official_complete_submission_url"],
        "official_primary_document_url": item["official_primary_document_url"],
        "source_record_sha256": item["source_record_sha256"],
        "selected_text_sha256": strip_exact_sha256_tag(
            item["raw_primary_document_sha256"]
        ),
        "normalized_text_sha256": strip_exact_sha256_tag(
            item["normalized_text_sha256"]
        ),
        "complete_response_sha256": strip_exact_sha256_tag(
            item["complete_response_sha256"]
        ),
        "compatibility_record_sha256": digest,
    }
    for key, child in expected.items():
        if key in record and record[key] != child:
            _reject("v319_contract_proof_compatibility_mismatch")
    return item


def validate_nullable_universe_event_proof(
    value: Any,
    *,
    universe: Mapping[str, Any],
    content_manifest: Mapping[str, Any],
    compatibility_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one proof against all three sealed private authorities."""

    universe_item = validate_nullable_universe_manifest(universe)
    content_item = validate_nullable_content_manifest(content_manifest)
    records = [validate_compatibility_record(child) for child in compatibility_records]
    if len(records) != DEVELOPMENT_DOCUMENT_COUNT:
        _reject("v319_contract_proof_compatibility_count")
    compatibility_by_hash = {canonical_sha256(child): child for child in records}
    compatibility_accessions = {
        child["accession_number"] for child in compatibility_by_hash.values()
    }
    universe_compatibility_hashes = {
        child["compatibility_record_sha256"] for child in universe_item["records"]
    }
    content_compatibility_hashes = {
        child["compatibility_record_sha256"]
        for child in content_item["documents"]
    }
    if (
        len(compatibility_by_hash) != DEVELOPMENT_DOCUMENT_COUNT
        or len(compatibility_accessions) != DEVELOPMENT_DOCUMENT_COUNT
        or universe_compatibility_hashes != set(compatibility_by_hash)
        or content_compatibility_hashes != set(compatibility_by_hash)
    ):
        _reject("v319_contract_proof_compatibility_duplicate")
    item = _strict_mapping(
        value,
        frozenset(UNIVERSE_EVENT_PROOF_FIELDS),
        code="v319_contract_universe_event_proof_shape",
    )
    validate_self_sha256(item, field="universe_event_proof_sha256")
    current = validate_nullable_universe_record(item["current_record"])
    current_content = validate_nullable_content_record(item["current_content_record"])
    accession = current["accession_number"]
    universe_matches = [
        child for child in universe_item["records"] if child["accession_number"] == accession
    ]
    content_matches = [
        child
        for child in content_item["documents"]
        if child["accession_number"] == accession
    ]
    if universe_matches != [current] or content_matches != [current_content]:
        _reject("v319_contract_proof_membership")
    shared = (
        "accession_number",
        "form",
        "availability_session",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "compatibility_record_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
    )
    if any(current[key] != current_content[key] for key in shared):
        _reject("v319_contract_proof_cross_manifest")
    compatibility = _compatibility_record_for_successor(
        current, compatibility_by_hash
    )
    _compatibility_record_for_successor(current_content, compatibility_by_hash)
    if (
        current_content["selected_text_bytes"] != compatibility["selected_text_length"]
        or current_content["normalized_text_bytes"]
        != compatibility["normalized_text_length"]
        or item["schema_version"]
        != NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
        or item["compatibility_manifest_sha256"]
        != universe_item["compatibility_manifest_sha256"]
        or item["compatibility_manifest_sha256"]
        != content_item["compatibility_manifest_sha256"]
        or item["universe_sha256"] != universe_item["universe_sha256"]
        or item["universe_sha256"] != content_item["universe_sha256"]
        or item["content_manifest_sha256"]
        != content_item["content_manifest_sha256"]
        or item["calendar_sessions_sha256"] != CALENDAR_SESSIONS_SHA256
        or item["current_record_sha256"] != canonical_sha256(current)
        or item["current_content_record_sha256"]
        != canonical_sha256(current_content)
        or item["current_filing_sha256"]
        != current_content["normalized_text_sha256"]
        or item["prior_selection"] != PRIOR_SELECTION_POLICY
    ):
        _reject("v319_contract_universe_event_proof_identity")
    ordered = list(universe_item["records"])
    current_index = ordered.index(current)
    expected_prior = next(
        (
            child
            for child in reversed(ordered[:current_index])
            if child["form"] == current["form"]
        ),
        None,
    )
    prior_values = (
        item["prior_same_form_record"],
        item["prior_same_form_record_sha256"],
        item["prior_same_form_content_record"],
        item["prior_same_form_content_record_sha256"],
        item["prior_same_form_filing_sha256"],
    )
    if expected_prior is None:
        if any(child is not None for child in prior_values):
            _reject("v319_contract_proof_prior")
    else:
        prior_record = validate_nullable_universe_record(
            item["prior_same_form_record"]
        )
        prior_content = validate_nullable_content_record(
            item["prior_same_form_content_record"]
        )
        expected_prior_content = next(
            child
            for child in content_item["documents"]
            if child["accession_number"] == expected_prior["accession_number"]
        )
        if (
            prior_record != expected_prior
            or prior_content != expected_prior_content
            or item["prior_same_form_record_sha256"]
            != canonical_sha256(prior_record)
            or item["prior_same_form_content_record_sha256"]
            != canonical_sha256(prior_content)
            or item["prior_same_form_filing_sha256"]
            != prior_content["normalized_text_sha256"]
        ):
            _reject("v319_contract_proof_prior")
        _compatibility_record_for_successor(prior_record, compatibility_by_hash)
        _compatibility_record_for_successor(prior_content, compatibility_by_hash)
    return item


def _validate_direction_sanitizer_source(source: Any) -> dict[str, Any]:
    """Validate the inherited event's complete pure structure and hashes."""

    fields = {
        "schema_version",
        "preprocessor_version",
        "input_scope",
        "current_filing_sha256",
        "prior_same_form_filing_sha256",
        "identity_lexicon_sha256",
        "sentences",
        "sentences_sha256",
        "model_payload",
        "model_payload_sha256",
        "redaction_report",
        "preprocessed_event_sha256",
    }
    if not isinstance(source, Mapping) or set(source) != fields:
        _reject("v319_direction_sanitizer_source_invalid")
    item = copy.deepcopy(dict(source))
    try:
        from .sec_filing_gemma_contract import (
            CANONICAL_IDENTITY_LEXICON,
            PREPROCESSOR_VERSION,
            build_extractor_model_payload,
        )
        from .sec_filing_gemma_preprocessor import (
            MAX_CURRENT_SENTENCES,
            MAX_INPUT_BYTES,
            MAX_PRIOR_SENTENCES,
            MAX_SENTENCE_CHARACTERS,
            MAX_SENTENCES,
            PREPROCESSED_EVENT_SCHEMA_VERSION,
            _canonical_ascii_source,
            _identity_patterns,
            _normalize_lexicon,
            _residual_counts,
        )

        normalized_lexicon = _normalize_lexicon(CANONICAL_IDENTITY_LEXICON)
        identity_patterns = _identity_patterns(normalized_lexicon)
        if (
            item["schema_version"] != PREPROCESSED_EVENT_SCHEMA_VERSION
            or item["preprocessor_version"] != PREPROCESSOR_VERSION
            or item["input_scope"]
            != "exact_current_and_optional_immediate_prior_same_form_only"
            or not _is_hex(item["current_filing_sha256"], 64)
            or (
                item["prior_same_form_filing_sha256"] is not None
                and not _is_hex(item["prior_same_form_filing_sha256"], 64)
            )
            or item["identity_lexicon_sha256"]
            != canonical_sha256(list(normalized_lexicon))
            or not _is_hex(item["sentences_sha256"], 64)
            or not _is_hex(item["model_payload_sha256"], 64)
            or not _is_hex(item["preprocessed_event_sha256"], 64)
        ):
            raise ValueError

        raw_sentences = item["sentences"]
        if type(raw_sentences) is not list or not 1 <= len(raw_sentences) <= MAX_SENTENCES:
            raise ValueError
        current_count = 0
        prior_count = 0
        prior_started = False
        sentences: list[dict[str, str]] = []
        for raw in raw_sentences:
            if type(raw) is not dict or set(raw) != {"id", "text"}:
                raise ValueError
            sentence_id = raw["id"]
            text = raw["text"]
            if (
                type(sentence_id) is not str
                or type(text) is not str
                or not text
                or text != text.strip()
                or not text.isascii()
                or len(text) > MAX_SENTENCE_CHARACTERS
                or _canonical_ascii_source(text) != text
            ):
                raise ValueError
            if sentence_id.startswith("C") and not prior_started:
                current_count += 1
                expected_id = f"C{current_count:04d}"
            elif sentence_id.startswith("P"):
                prior_started = True
                prior_count += 1
                expected_id = f"P{prior_count:04d}"
            else:
                raise ValueError
            if sentence_id != expected_id:
                raise ValueError
            sentences.append({"id": sentence_id, "text": text})
        has_prior = item["prior_same_form_filing_sha256"] is not None
        utf8_bytes = len(
            "\n".join(sentence["text"] for sentence in sentences).encode("utf-8")
        )
        if (
            current_count < 1
            or current_count > MAX_CURRENT_SENTENCES
            or prior_count > MAX_PRIOR_SENTENCES
            or (prior_count > 0) is not has_prior
            or utf8_bytes > MAX_INPUT_BYTES
            or any(_residual_counts(sentences, identity_patterns))
        ):
            raise ValueError

        report = item["redaction_report"]
        expected_report = {
            "identity_matches_remaining": 0,
            "numeric_matches_remaining": 0,
            "date_matches_remaining": 0,
            "forbidden_context_fields": 0,
            "sentence_count": len(sentences),
            "utf8_bytes": utf8_bytes,
            "maximum_sentence_characters": max(
                len(sentence["text"]) for sentence in sentences
            ),
        }
        rebuilt_payload = build_extractor_model_payload(sentences)
        unsigned = {
            key: copy.deepcopy(item[key])
            for key in fields
            if key != "preprocessed_event_sha256"
        }
        if (
            type(report) is not dict
            or set(report) != set(expected_report)
            or any(
                type(report[field]) is not int or report[field] < 0
                for field in expected_report
            )
            or report != expected_report
            or item["sentences_sha256"] != canonical_sha256(sentences)
            or canonical_json_bytes(item["model_payload"])
            != canonical_json_bytes(rebuilt_payload)
            or item["model_payload_sha256"] != canonical_sha256(rebuilt_payload)
            or item["preprocessed_event_sha256"] != canonical_sha256(unsigned)
        ):
            raise ValueError
    except ContractViolation:
        _reject("v319_direction_sanitizer_source_invalid")
    except Exception:
        _reject("v319_direction_sanitizer_source_invalid")
    return item


def build_direction_sanitized_preprocessed_event(source: Any) -> dict[str, Any]:
    """Remove exactly the frozen security-direction sentences and rebind bytes."""

    source_item = _validate_direction_sanitizer_source(source)
    retained_current: list[dict[str, str]] = []
    retained_prior: list[dict[str, str]] = []
    removed_current = 0
    removed_prior = 0
    for sentence in source_item["sentences"]:
        is_prior = sentence["id"].startswith("P")
        if _blinded_text_has_security_direction(sentence):
            if is_prior:
                removed_prior += 1
            else:
                removed_current += 1
            continue
        target = retained_prior if is_prior else retained_current
        prefix = "P" if is_prior else "C"
        target.append({"id": f"{prefix}{len(target) + 1:04d}", "text": sentence["text"]})
    if not retained_current:
        _reject("v319_direction_sanitizer_empty_current")
    if source_item["prior_same_form_filing_sha256"] is not None and not retained_prior:
        _reject("v319_direction_sanitizer_empty_prior")

    sentences = [*retained_current, *retained_prior]
    try:
        from .sec_filing_gemma_contract import build_extractor_model_payload

        # The retained sequence enters this builder exactly once in this invocation.
        model_payload = build_extractor_model_payload(sentences)
    except ContractViolation:
        raise
    except Exception:
        _reject("v319_direction_sanitizer_source_invalid")
    body = {
        "schema_version": DIRECTION_SANITIZED_EVENT_SCHEMA_VERSION,
        "sanitizer_schema_version": DIRECTION_SANITIZER_SCHEMA_VERSION,
        "source_preprocessed_event_sha256": source_item[
            "preprocessed_event_sha256"
        ],
        "source_sentence_count": len(source_item["sentences"]),
        "removed_current_sentence_count": removed_current,
        "removed_prior_sentence_count": removed_prior,
        "retained_current_sentence_count": len(retained_current),
        "retained_prior_sentence_count": len(retained_prior),
        "sentences": sentences,
        "sentences_sha256": canonical_sha256(sentences),
        "model_payload": model_payload,
        "model_payload_sha256": canonical_sha256(model_payload),
    }
    return {
        **body,
        "preprocessed_event_sha256": canonical_sha256(body),
    }


def _validate_direction_sanitized_event_shape(candidate: Any) -> dict[str, Any]:
    """Independently validate one sanitized event without source reconstruction."""

    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != set(DIRECTION_SANITIZED_EVENT_FIELDS)
    ):
        raise ValueError
    item = copy.deepcopy(dict(candidate))
    count_fields = (
        "source_sentence_count",
        "removed_current_sentence_count",
        "removed_prior_sentence_count",
        "retained_current_sentence_count",
        "retained_prior_sentence_count",
    )
    if (
        item["schema_version"] != DIRECTION_SANITIZED_EVENT_SCHEMA_VERSION
        or item["sanitizer_schema_version"] != DIRECTION_SANITIZER_SCHEMA_VERSION
        or not _is_hex(item["source_preprocessed_event_sha256"], 64)
        or not _is_hex(item["sentences_sha256"], 64)
        or not _is_hex(item["model_payload_sha256"], 64)
        or not _is_hex(item["preprocessed_event_sha256"], 64)
        or any(type(item[field]) is not int or item[field] < 0 for field in count_fields)
    ):
        raise ValueError
    sentences = item["sentences"]
    if type(sentences) is not list:
        raise ValueError
    current_count = 0
    prior_count = 0
    prior_started = False
    for sentence in sentences:
        if type(sentence) is not dict or set(sentence) != {"id", "text"}:
            raise ValueError
        sentence_id = sentence["id"]
        text = sentence["text"]
        if (
            type(sentence_id) is not str
            or type(text) is not str
            or not text
            or text != text.strip()
            or not text.isascii()
            or _blinded_text_has_security_direction(sentence)
        ):
            raise ValueError
        if sentence_id.startswith("C") and not prior_started:
            current_count += 1
            expected_id = f"C{current_count:04d}"
        elif sentence_id.startswith("P"):
            prior_started = True
            prior_count += 1
            expected_id = f"P{prior_count:04d}"
        else:
            raise ValueError
        if sentence_id != expected_id:
            raise ValueError
    if (
        current_count < 1
        or current_count != item["retained_current_sentence_count"]
        or prior_count != item["retained_prior_sentence_count"]
        or item["source_sentence_count"]
        != item["removed_current_sentence_count"]
        + item["removed_prior_sentence_count"]
        + current_count
        + prior_count
        or (item["removed_prior_sentence_count"] > 0 and prior_count == 0)
        or item["sentences_sha256"] != canonical_sha256(sentences)
    ):
        raise ValueError
    from .sec_filing_gemma_contract import build_extractor_model_payload

    rebuilt_payload = build_extractor_model_payload(sentences)
    unsigned = {
        key: copy.deepcopy(item[key])
        for key in DIRECTION_SANITIZED_EVENT_FIELDS
        if key != "preprocessed_event_sha256"
    }
    if (
        canonical_json_bytes(item["model_payload"])
        != canonical_json_bytes(rebuilt_payload)
        or item["model_payload_sha256"] != canonical_sha256(rebuilt_payload)
        or item["preprocessed_event_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError
    return item


def validate_direction_sanitized_preprocessed_event(
    source: Any,
    candidate: Any,
    *,
    expected_source_preprocessed_event_sha256: str,
) -> dict[str, Any]:
    """Require the exact deterministic source-to-sanitized derivation."""

    try:
        if not _is_hex(expected_source_preprocessed_event_sha256, 64):
            raise ValueError
        source_item = _validate_direction_sanitizer_source(source)
        if (
            source_item["preprocessed_event_sha256"]
            != expected_source_preprocessed_event_sha256
        ):
            raise ValueError
        expected = build_direction_sanitized_preprocessed_event(source_item)
        validated = _validate_direction_sanitized_event_shape(candidate)
        if canonical_json_bytes(validated) != canonical_json_bytes(expected):
            raise ValueError
        return copy.deepcopy(validated)
    except Exception as exc:
        raise ContractViolation("v319_direction_sanitized_event_invalid") from exc


def _blinded_request_identity_tokens(proof: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every private source/proof token forbidden inside model bytes."""

    tokens: set[str] = set()
    for field in _BLINDED_REQUEST_PROOF_IDENTITY_FIELDS:
        value = proof.get(field)
        if type(value) is str and value:
            tokens.add(value)
    for field in (
        "current_record",
        "current_content_record",
        "prior_same_form_record",
        "prior_same_form_content_record",
    ):
        record = proof.get(field)
        if not isinstance(record, Mapping):
            continue
        for identity_field in _BLINDED_REQUEST_RECORD_IDENTITY_FIELDS:
            value = record.get(identity_field)
            if type(value) is str and value:
                tokens.add(value)
    return tuple(sorted(tokens, key=lambda value: value.encode("utf-8")))


def _normalized_request_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _is_structured_source_missingness_key(value: str) -> bool:
    normalized = _normalized_request_key(value)
    if not normalized:
        return False
    if normalized in _BLINDED_REQUEST_RECORD_IDENTITY_FIELDS:
        # A locator/source key with a null/Boolean value is itself an explicit
        # structured identity or missingness channel.
        return True
    parts = frozenset(part for part in normalized.split("_") if part)
    if "missing" in parts or "missingness" in parts:
        return True
    missingness_words = {"absent", "available", "exists", "has", "null", "present"}
    return bool(parts & missingness_words) and any(
        subject in normalized for subject in _BLINDED_REQUEST_MISSINGNESS_SUBJECTS
    )


def _blinded_text_has_security_direction(sentence: Mapping[str, Any]) -> bool:
    """Apply the one frozen V3.19 span-local security-direction predicate."""

    folded = sentence["text"].casefold()
    masked = _BLINDED_TEXT_MARKET_SHARE_SPAN_RE.sub(";", folded)
    remove = _BLINDED_TEXT_SECURITY_DIRECTION_RE.search(masked) is not None
    return remove


def _blinded_text_contains_forbidden_context(value: str) -> bool:
    """Reject identity, time, outcome, or source-state clues in sentence text."""

    # The frozen preprocessor emits canonical ASCII. Rejecting any non-ASCII
    # sentence here closes homoglyph, zero-width, and full-width bypasses while
    # leaving the fixed outer prompt and schema untouched.
    if not value.isascii():
        return True
    folded = value.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if not normalized:
        return False
    security_direction = _blinded_text_has_security_direction({"text": value})
    words = frozenset(normalized.split())
    return_outcome = bool(
        words & {"return", "returns"}
        and words
        & (
            _BLINDED_TEXT_MARKET_DIRECTION_WORDS
            | {"negative", "neutral", "positive", "performance"}
        )
    )
    buy_hold_comparison = bool(
        {"buy", "hold"} <= words
        and words & {"benchmark", "comparison", "performance", "return", "strategy"}
    )
    return (
        _BLINDED_TEXT_ISSUER_RE.search(folded) is not None
        or _BLINDED_TEXT_SOURCE_VOCABULARY_RE.search(folded) is not None
        or _BLINDED_TEXT_SOURCE_MISSINGNESS_RE.search(normalized) is not None
        or any(pattern.search(folded) is not None for pattern in _BLINDED_TEXT_EXACT_DATE_RES)
        or any(
            pattern.search(normalized) is not None
            for pattern in _BLINDED_TEXT_MARKET_CONTEXT_RES
        )
        or _BLINDED_TEXT_KEYED_LABEL_RE.search(folded) is not None
        or _BLINDED_TEXT_MODEL_ACTION_RE.search(folded) is not None
        or security_direction
        or return_outcome
        or buy_hold_comparison
    )


def _blinded_text_contains_encoded_private_context(
    value: str,
    identity_tokens: Sequence[str],
) -> bool:
    """Decode self-delimiting base64 tokens and rescan their readable text."""

    for match in _BLINDED_TEXT_BASE64_TOKEN_RE.finditer(value):
        token = match.group(0)
        unpadded = token.rstrip("=")
        padding = "=" * ((4 - len(unpadded) % 4) % 4)
        try:
            decoded_bytes = base64.b64decode(
                (unpadded + padding).encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
            decoded = decoded_bytes.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError):
            continue
        if not decoded.isascii() or not decoded.isprintable():
            continue
        folded = decoded.casefold()
        if any(token_value.casefold() in folded for token_value in identity_tokens):
            return True
        if _blinded_text_contains_forbidden_context(decoded):
            return True
    return False


def _exact_blinded_sentences(
    decoded: Any,
    *,
    request_bytes: bytes,
    supplied_sentence_ids: Sequence[str],
    has_prior: bool,
) -> tuple[dict[str, str], ...] | None:
    """Recover only the exact frozen canonical sentence payload."""

    try:
        from .sec_filing_gemma_contract import build_extractor_model_payload
        from .sec_filing_gemma_preprocessor import (
            MAX_CURRENT_SENTENCES,
            MAX_INPUT_BYTES,
            MAX_PRIOR_SENTENCES,
            MAX_SENTENCE_CHARACTERS,
            MAX_SENTENCES,
        )

        if type(decoded) is not dict:
            return None
        messages = decoded.get("messages")
        if type(messages) is not list or len(messages) != 2:
            return None
        user_message = messages[1]
        if type(user_message) is not dict or set(user_message) != {"role", "content"}:
            return None
        user_content = user_message.get("content")
        if user_message.get("role") != "user" or type(user_content) is not str:
            return None
        user_payload = json.loads(user_content)
        if type(user_payload) is not dict or set(user_payload) != {"sentences"}:
            return None
        raw_sentences = user_payload["sentences"]
        if (
            type(raw_sentences) is not list
            or not 1 <= len(raw_sentences) <= MAX_SENTENCES
        ):
            return None
        sentences: list[dict[str, str]] = []
        current_count = 0
        prior_count = 0
        prior_started = False
        for raw in raw_sentences:
            if type(raw) is not dict or set(raw) != {"id", "text"}:
                return None
            sentence_id = raw["id"]
            text = raw["text"]
            if (
                type(sentence_id) is not str
                or type(text) is not str
                or not text
                or text != text.strip()
                or not text.isascii()
                or len(text) > MAX_SENTENCE_CHARACTERS
            ):
                return None
            if sentence_id.startswith("C") and not prior_started:
                current_count += 1
                expected_id = f"C{current_count:04d}"
            elif sentence_id.startswith("P"):
                prior_started = True
                prior_count += 1
                expected_id = f"P{prior_count:04d}"
            else:
                return None
            if sentence_id != expected_id:
                return None
            sentences.append({"id": sentence_id, "text": text})
        if (
            current_count < 1
            or current_count > MAX_CURRENT_SENTENCES
            or prior_count > MAX_PRIOR_SENTENCES
            or (prior_count > 0) is not has_prior
            or tuple(supplied_sentence_ids)
            != tuple(sentence["id"] for sentence in sentences)
            or len("\n".join(sentence["text"] for sentence in sentences).encode("utf-8"))
            > MAX_INPUT_BYTES
        ):
            return None
        rebuilt = build_extractor_model_payload(sentences)
        if canonical_json_bytes(rebuilt) != request_bytes:
            return None
        return tuple(sentences)
    except Exception:
        return None


def _sentences_violate_frozen_preprocessor(
    sentences: Sequence[Mapping[str, str]],
    *,
    identity_tokens: Sequence[str],
) -> bool:
    """Reapply the exact inherited residual rules at the hard request gate."""

    try:
        from .sec_filing_gemma_contract import CANONICAL_IDENTITY_LEXICON
        from .sec_filing_gemma_preprocessor import (
            _canonical_ascii_source,
            _identity_patterns,
            _normalize_lexicon,
            _residual_counts,
        )

        patterns = _identity_patterns(_normalize_lexicon(CANONICAL_IDENTITY_LEXICON))
        if any(
            _canonical_ascii_source(sentence["text"]) != sentence["text"]
            for sentence in sentences
        ):
            return True
        if any(_residual_counts(sentences, patterns)):
            return True
    except Exception:
        return True
    return any(
        _blinded_text_contains_forbidden_context(sentence["text"])
        or _blinded_text_contains_encoded_private_context(
            sentence["text"], identity_tokens
        )
        for sentence in sentences
    )


def _request_json_contains_private_structure(
    value: Any,
    *,
    identity_tokens: Sequence[str],
    blinded_text: bool = False,
    depth: int = 0,
) -> bool:
    """Inspect the outer payload and canonical JSON embedded in user content."""

    if depth > 64:
        return True
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str or _is_structured_source_missingness_key(key):
                return True
            if _request_json_contains_private_structure(
                child,
                identity_tokens=identity_tokens,
                blinded_text=blinded_text,
                depth=depth + 1,
            ):
                return True
        return False
    if isinstance(value, list):
        return any(
            _request_json_contains_private_structure(
                child,
                identity_tokens=identity_tokens,
                blinded_text=blinded_text,
                depth=depth + 1,
            )
            for child in value
        )
    if type(value) is not str:
        return False
    folded = value.casefold()
    if any(token.casefold() in folded for token in identity_tokens):
        return True
    if blinded_text and _blinded_text_contains_forbidden_context(value):
        return True
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        nested = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return False
    if not isinstance(nested, (dict, list)):
        return False
    nested_blinded_text = blinded_text or (
        isinstance(nested, Mapping) and set(nested) == {"sentences"}
    )
    return _request_json_contains_private_structure(
        nested,
        identity_tokens=identity_tokens,
        blinded_text=nested_blinded_text,
        depth=depth + 1,
    )


def _blinded_request_contains_private_identity(
    request_bytes: bytes,
    proof: Mapping[str, Any],
    *,
    supplied_sentence_ids: Sequence[str],
) -> bool:
    identity_tokens = _blinded_request_identity_tokens(proof)
    folded_bytes = request_bytes.lower()
    if any(token.encode("utf-8").lower() in folded_bytes for token in identity_tokens):
        return True
    try:
        decoded = json.loads(request_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        # This validator is itself a hard gate. Malformed bytes cannot be
        # treated as privacy-safe merely because the shared shape validator is
        # expected to run later.
        return True
    sentences = _exact_blinded_sentences(
        decoded,
        request_bytes=request_bytes,
        supplied_sentence_ids=supplied_sentence_ids,
        has_prior=proof.get("prior_same_form_record") is not None,
    )
    if sentences is None or _sentences_violate_frozen_preprocessor(
        sentences,
        identity_tokens=identity_tokens,
    ):
        return True
    return _request_json_contains_private_structure(
        decoded,
        identity_tokens=identity_tokens,
        blinded_text=False,
    )


def validate_blinded_model_request(
    value: Any,
    *,
    proof: Mapping[str, Any],
    sanitized_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the V3.19 wrapper after the complete proof has passed."""

    try:
        validated_sanitized = _validate_direction_sanitized_event_shape(
            sanitized_event
        )
        if not isinstance(value, Mapping):
            raise ValueError
        request_bytes_for_binding = value.get("request_bytes")
        ordered_ids = [
            sentence["id"] for sentence in validated_sanitized["sentences"]
        ]
        if (
            value.get("preprocessed_event_sha256")
            != validated_sanitized["preprocessed_event_sha256"]
            or value.get("supplied_sentence_ids") != ordered_ids
            or type(request_bytes_for_binding) is not bytes
            or request_bytes_for_binding
            != canonical_json_bytes(validated_sanitized["model_payload"])
            or value.get("request_sha256")
            != sha256_bytes(request_bytes_for_binding)
        ):
            raise ValueError
    except Exception as exc:
        raise ContractViolation(
            "v319_contract_blinded_request_sanitized_event"
        ) from exc

    if not isinstance(value, Mapping) or set(value) != set(BLINDED_MODEL_REQUEST_FIELDS):
        _reject("v319_contract_blinded_request_shape")
    item = dict(value)
    if (
        not isinstance(proof, Mapping)
        or set(proof) != set(UNIVERSE_EVENT_PROOF_FIELDS)
    ):
        _reject("v319_contract_blinded_request_proof")
    proof_item = validate_self_sha256(
        proof, field="universe_event_proof_sha256"
    )
    current = validate_nullable_universe_record(proof_item["current_record"])
    if (
        proof_item["schema_version"]
        != NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
        or proof_item["current_record_sha256"] != canonical_sha256(current)
    ):
        _reject("v319_contract_blinded_request_proof")
    request_bytes = item["request_bytes"]
    sentence_ids = item["supplied_sentence_ids"]
    if (
        item["schema_version"] != BLINDED_MODEL_REQUEST_SCHEMA_VERSION
        or item["accession_number"] != current.get("accession_number")
        or item["form"] != current.get("form")
        or item["availability_session"] != current.get("availability_session")
        or not _is_hex(item["preprocessed_event_sha256"], 64)
        or type(sentence_ids) is not list
        or not sentence_ids
        or any(type(child) is not str or not child for child in sentence_ids)
        or len(sentence_ids) != len(set(sentence_ids))
        or type(request_bytes) is not bytes
        or not 0 < len(request_bytes) <= MODEL_REQUEST_MAX_BYTES
        or item["request_sha256"] != sha256_bytes(request_bytes)
    ):
        _reject("v319_contract_blinded_request_identity")
    prior = proof_item["prior_same_form_record"]
    if prior is not None:
        prior = validate_nullable_universe_record(prior)
    if _blinded_request_contains_private_identity(
        request_bytes,
        proof_item,
        supplied_sentence_ids=sentence_ids,
    ):
        _reject("v319_contract_blinded_request_private_identity")
    return copy.deepcopy(item)


def build_nullable_model_slice(
    model_requests: Sequence[Mapping[str, Any]],
    universe_event_proofs: Sequence[Mapping[str, Any]],
    source_preprocessed_events: Sequence[Mapping[str, Any]],
    direction_sanitized_preprocessed_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact request-bytes-excluding V3.19 model-slice commitment."""

    if (
        not isinstance(model_requests, Sequence)
        or isinstance(model_requests, (str, bytes, bytearray))
        or not isinstance(universe_event_proofs, Sequence)
        or isinstance(universe_event_proofs, (str, bytes, bytearray))
        or not isinstance(source_preprocessed_events, Sequence)
        or isinstance(source_preprocessed_events, (str, bytes, bytearray))
        or not isinstance(direction_sanitized_preprocessed_events, Sequence)
        or isinstance(
            direction_sanitized_preprocessed_events, (str, bytes, bytearray)
        )
        or len(model_requests) != DEVELOPMENT_DOCUMENT_COUNT
        or len(universe_event_proofs) != DEVELOPMENT_DOCUMENT_COUNT
        or len(source_preprocessed_events) != DEVELOPMENT_DOCUMENT_COUNT
        or len(direction_sanitized_preprocessed_events)
        != DEVELOPMENT_DOCUMENT_COUNT
    ):
        _reject("v319_contract_model_slice_count")
    requests: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    source_events: list[dict[str, Any]] = []
    sanitized_events: list[dict[str, Any]] = []
    accessions: list[str] = []
    for request, proof, source_event, sanitized_event in zip(
        model_requests,
        universe_event_proofs,
        source_preprocessed_events,
        direction_sanitized_preprocessed_events,
        strict=True,
    ):
        if (
            not isinstance(request, Mapping)
            or not isinstance(proof, Mapping)
            or not isinstance(source_event, Mapping)
            or not isinstance(sanitized_event, Mapping)
        ):
            _reject("v319_contract_model_slice_member")
        source_copy = _validate_direction_sanitizer_source(source_event)
        validated_sanitized = validate_direction_sanitized_preprocessed_event(
            source_copy,
            sanitized_event,
            expected_source_preprocessed_event_sha256=source_copy[
                "preprocessed_event_sha256"
            ],
        )
        validated_request = validate_blinded_model_request(
            request,
            proof=proof,
            sanitized_event=validated_sanitized,
        )
        proof_copy = copy.deepcopy(dict(proof))
        current = proof_copy["current_record"]
        if (
            validated_request["accession_number"] != current["accession_number"]
            or validated_request["form"] != current["form"]
            or validated_request["availability_session"]
            != current["availability_session"]
            or source_copy["current_filing_sha256"]
            != proof_copy["current_filing_sha256"]
            or source_copy["prior_same_form_filing_sha256"]
            != proof_copy["prior_same_form_filing_sha256"]
        ):
            _reject("v319_contract_model_slice_order")
        requests.append(validated_request)
        proofs.append(proof_copy)
        source_events.append(source_copy)
        sanitized_events.append(validated_sanitized)
        accessions.append(validated_request["accession_number"])
    if len(set(accessions)) != DEVELOPMENT_DOCUMENT_COUNT:
        _reject("v319_contract_model_slice_order")
    index = [
        {key: copy.deepcopy(child[key]) for key in child if key != "request_bytes"}
        for child in requests
    ]
    hash_body = {
        "stage": "development",
        "attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "model_requests": index,
        "universe_event_proofs": proofs,
        "source_preprocessed_event_sha256s": [
            event["preprocessed_event_sha256"] for event in source_events
        ],
        "direction_sanitized_preprocessed_event_sha256s": [
            event["preprocessed_event_sha256"] for event in sanitized_events
        ],
        "direction_sanitizer_partition_removal_counts": [
            {
                "removed_current_sentence_count": event[
                    "removed_current_sentence_count"
                ],
                "removed_prior_sentence_count": event[
                    "removed_prior_sentence_count"
                ],
            }
            for event in sanitized_events
        ],
    }
    return {
        "stage": "development",
        "attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "model_requests": requests,
        "universe_event_proofs": proofs,
        "source_preprocessed_events": source_events,
        "direction_sanitized_preprocessed_events": sanitized_events,
        "model_slice_sha256": canonical_sha256(hash_body),
    }


def validate_nullable_model_slice(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "stage",
        "attempt_id",
        "model_requests",
        "universe_event_proofs",
        "source_preprocessed_events",
        "direction_sanitized_preprocessed_events",
        "model_slice_sha256",
    }:
        _reject("v319_contract_model_slice_shape")
    item = dict(value)
    rebuilt = build_nullable_model_slice(
        item["model_requests"],
        item["universe_event_proofs"],
        item["source_preprocessed_events"],
        item["direction_sanitized_preprocessed_events"],
    )
    if item != rebuilt:
        _reject("v319_contract_model_slice_identity")
    return copy.deepcopy(rebuilt)


def project_science_manifest(source_manifest: Any) -> dict[str, Any]:
    """Select exactly the twelve scientific keys from a source manifest."""

    if not isinstance(source_manifest, Mapping):
        _reject("v319_contract_science_source_shape")
    if any(key not in source_manifest for key in SCIENCE_PROJECTION_KEYS):
        _reject("v319_contract_science_source_keys")
    return {
        key: copy.deepcopy(source_manifest[key]) for key in SCIENCE_PROJECTION_KEYS
    }


def verify_frozen_science_projection(value: Any) -> dict[str, Any]:
    """Accept only the exact 38,320-byte frozen scientific projection."""

    if not isinstance(value, Mapping) or set(value) != set(SCIENCE_PROJECTION_KEYS):
        _reject("v319_contract_science_projection_keys")
    detached = {key: copy.deepcopy(value[key]) for key in SCIENCE_PROJECTION_KEYS}
    encoded = canonical_json_bytes(detached)
    if len(encoded) != SCIENCE_PROJECTION_BYTE_COUNT:
        _reject("v319_contract_science_projection_size")
    if sha256_bytes(encoded) != SCIENCE_PROJECTION_SHA256:
        _reject("v319_contract_science_projection_hash")
    return detached


def build_frozen_science_projection() -> dict[str, Any]:
    """Build and authenticate the frozen projection through the pinned adapter."""

    # This import is pure: the predecessor contract has its own literal
    # self-check and performs no external I/O.
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        CONTRACT_SHA256 as predecessor_internal_sha256,
        build_contract_manifest as build_predecessor_manifest,
    )

    if predecessor_internal_sha256 != SCIENTIFIC_CONTRACT_INTERNAL_SHA256:
        _reject("v319_contract_science_parent_internal")
    return verify_frozen_science_projection(
        project_science_manifest(build_predecessor_manifest())
    )


def build_source_authority_pins() -> dict[str, Any]:
    """Return a detached public description of all frozen v3.8 authority pins."""

    return {
        "base": {
            "commit": V38_SOURCE_COMMIT,
            "tree": V38_SOURCE_TREE,
            "parent": V38_SOURCE_PARENT,
        },
        "terminal": {
            "path": V38_TERMINAL_PATH,
            "git_blob_sha1": V38_TERMINAL_BLOB_SHA1,
            "literal_sha256": V38_TERMINAL_LITERAL_SHA256,
            "internal_sha256": V38_TERMINAL_INTERNAL_SHA256,
        },
        "source_authority": {
            "path": V38_SOURCE_AUTHORITY_PATH,
            "git_blob_sha1": V38_SOURCE_AUTHORITY_BLOB_SHA1,
            "literal_sha256": V38_SOURCE_AUTHORITY_LITERAL_SHA256,
        },
        "modules": {
            "source": {
                "path": V38_SOURCE_MODULE_PATH,
                "git_blob_sha1": V38_SOURCE_MODULE_GIT_BLOB_SHA1,
                "literal_sha256": V38_SOURCE_MODULE_LITERAL_SHA256,
            },
            "acquisition": {
                "path": V38_ACQUISITION_MODULE_PATH,
                "git_blob_sha1": V38_ACQUISITION_MODULE_GIT_BLOB_SHA1,
                "literal_sha256": V38_ACQUISITION_MODULE_LITERAL_SHA256,
            },
        },
        "private_aggregates": {
            "checkpoint_file_sha256": V38_CHECKPOINT_FILE_SHA256,
            "logical_checkpoint_sha256": V38_LOGICAL_CHECKPOINT_SHA256,
            "stage_source_seal_sha256": V38_STAGE_SOURCE_SEAL_SHA256,
            "compact_replay_sha256": V38_COMPACT_REPLAY_SHA256,
            "role_manifest_inventory_sha256": V38_ROLE_MANIFEST_INVENTORY_SHA256,
            "role_plan_sha256": V38_ROLE_PLAN_SHA256,
            "role_counts": dict(V38_ROLE_COUNTS),
            "v38_sec_requests": V38_SEC_REQUEST_COUNT,
            "experiment_family_sec_requests": EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
            "inventory_file_count": V38_INVENTORY_FILE_COUNT,
            "inventory_byte_count": V38_INVENTORY_BYTE_COUNT,
            "inventory_sha256": V38_INVENTORY_SHA256,
        },
    }


def build_v39_inherited_preregistration_pins() -> dict[str, Any]:
    """Return detached rejected-v3.9 provenance, never v3.19 authority."""

    return {
        "commit": V39_INHERITED_PREREG_COMMIT,
        "tree": V39_INHERITED_PREREG_TREE,
        "parent": V39_INHERITED_PREREG_PARENT,
        "path": V39_INHERITED_PREREG_PATH,
        "git_blob_sha1": V39_INHERITED_PREREG_GIT_BLOB_SHA1,
        "literal_sha256": V39_INHERITED_PREREG_LITERAL_SHA256,
        "literal_bytes": V39_INHERITED_PREREG_LITERAL_BYTES,
        "execution_authority": False,
    }


def build_v316_rejection_pins() -> dict[str, Any]:
    """Return immutable V3.16 P/I/F/X evidence, never V3.19 authority."""

    return {
        "preregistration": {
            "commit": V316_PREREGISTRATION_COMMIT,
            "tree": V316_PREREGISTRATION_TREE,
            "path": V316_PREREGISTRATION_PATH,
            "git_blob_sha1": V316_PREREGISTRATION_GIT_BLOB_SHA1,
            "literal_sha256": V316_PREREGISTRATION_LITERAL_SHA256,
            "literal_bytes": V316_PREREGISTRATION_LITERAL_BYTES,
        },
        "implementation": {
            "commit": V316_IMPLEMENTATION_COMMIT,
            "tree": V316_IMPLEMENTATION_TREE,
            "parent": V316_PREREGISTRATION_COMMIT,
        },
        "preflight_failure": {
            "commit": V316_PREFLIGHT_FAILURE_COMMIT,
            "tree": V316_PREFLIGHT_FAILURE_TREE,
            "parent": V316_IMPLEMENTATION_COMMIT,
            "path": V316_PREFLIGHT_FAILURE_PATH,
            "git_blob_sha1": V316_PREFLIGHT_FAILURE_GIT_BLOB_SHA1,
            "literal_sha256": V316_PREFLIGHT_FAILURE_LITERAL_SHA256,
            "literal_bytes": V316_PREFLIGHT_FAILURE_LITERAL_BYTES,
            "internal_sha256": V316_PREFLIGHT_FAILURE_INTERNAL_SHA256,
            "failure_code": "nullable_model_request_invalid",
            "preflight_consumed": True,
            "rerun_authorized": False,
            "development_authorized": False,
            "confirmation_and_final_opened": False,
            "real_money_authorized": False,
        },
        "rejection": {
            "commit": V316_REJECTION_COMMIT,
            "tree": V316_REJECTION_TREE,
            "parent": V316_PREFLIGHT_FAILURE_COMMIT,
            "path": V316_REJECTION_PATH,
            "git_blob_sha1": V316_REJECTION_GIT_BLOB_SHA1,
            "literal_sha256": V316_REJECTION_LITERAL_SHA256,
            "literal_bytes": V316_REJECTION_LITERAL_BYTES,
        },
        "phase1_node_count": 325,
        "phase1_case_sensitive_unique_node_count": 325,
        "phase1_duplicate_node_count": 0,
        "phase1_required_node_count": 24,
        "official_qualification_run_count": 1,
        "preflight_consumed": True,
        "external_effect_count": 0,
        "execution_authority": False,
    }


def build_v317_rejection_pins() -> dict[str, Any]:
    """Return immutable document-only V3.17 P/X evidence."""

    return {
        "preregistration": {
            "commit": V317_PREREGISTRATION_COMMIT,
            "tree": V317_PREREGISTRATION_TREE,
            "parent": V316_REJECTION_COMMIT,
            "path": V317_PREREGISTRATION_PATH,
            "git_blob_sha1": V317_PREREGISTRATION_GIT_BLOB_SHA1,
            "literal_sha256": V317_PREREGISTRATION_LITERAL_SHA256,
            "literal_bytes": V317_PREREGISTRATION_LITERAL_BYTES,
        },
        "rejection": {
            "commit": V317_REJECTION_COMMIT,
            "tree": V317_REJECTION_TREE,
            "parent": V317_PREREGISTRATION_COMMIT,
            "changed_paths": {
                V317_REJECTION_DOCUMENT_PATH: "A",
                V317_REJECTION_COMPARISON_PATH: "M",
            },
            "document": {
                "path": V317_REJECTION_DOCUMENT_PATH,
                "git_blob_sha1": V317_REJECTION_DOCUMENT_GIT_BLOB_SHA1,
                "literal_sha256": V317_REJECTION_DOCUMENT_LITERAL_SHA256,
                "literal_bytes": V317_REJECTION_DOCUMENT_LITERAL_BYTES,
            },
            "comparison": {
                "path": V317_REJECTION_COMPARISON_PATH,
                "git_blob_sha1": V317_REJECTION_COMPARISON_GIT_BLOB_SHA1,
                "literal_sha256": V317_REJECTION_COMPARISON_LITERAL_SHA256,
                "literal_bytes": V317_REJECTION_COMPARISON_LITERAL_BYTES,
            },
        },
        "implementation_exists": False,
        "preflight_artifact_exists": False,
        "private_authority_exists": False,
        "official_preflight_run_count": 0,
        "external_effect_count": 0,
        "execution_authority": False,
    }


def build_v318_rejection_pins() -> dict[str, Any]:
    """Return immutable document-only V3.18 P/X evidence."""

    return {
        "preregistration": {
            "commit": V318_PREREGISTRATION_COMMIT,
            "tree": V318_PREREGISTRATION_TREE,
            "parent": V317_REJECTION_COMMIT,
            "path": V318_PREREGISTRATION_PATH,
            "git_blob_sha1": V318_PREREGISTRATION_GIT_BLOB_SHA1,
            "literal_sha256": V318_PREREGISTRATION_LITERAL_SHA256,
            "literal_bytes": V318_PREREGISTRATION_LITERAL_BYTES,
        },
        "rejection": {
            "commit": V318_REJECTION_COMMIT,
            "tree": V318_REJECTION_TREE,
            "parent": V318_PREREGISTRATION_COMMIT,
            "changed_paths": {
                V318_REJECTION_DOCUMENT_PATH: "A",
                V318_REJECTION_COMPARISON_PATH: "M",
            },
            "document": {
                "path": V318_REJECTION_DOCUMENT_PATH,
                "git_blob_sha1": V318_REJECTION_DOCUMENT_GIT_BLOB_SHA1,
                "literal_sha256": V318_REJECTION_DOCUMENT_LITERAL_SHA256,
                "literal_bytes": V318_REJECTION_DOCUMENT_LITERAL_BYTES,
            },
            "comparison": {
                "path": V318_REJECTION_COMPARISON_PATH,
                "git_blob_sha1": V318_REJECTION_COMPARISON_GIT_BLOB_SHA1,
                "literal_sha256": V318_REJECTION_COMPARISON_LITERAL_SHA256,
                "literal_bytes": V318_REJECTION_COMPARISON_LITERAL_BYTES,
            },
        },
        "implementation_exists": False,
        "preflight_artifact_exists": False,
        "private_authority_exists": False,
        "official_preflight_run_count": 0,
        "external_effect_count": 0,
        "execution_authority": False,
    }


def build_qualification_contract() -> dict[str, Any]:
    """Return the exact dependency-scoped qualification authority."""

    return {
        "suite_id": QUALIFICATION_SUITE_ID,
        "phases": list(QUALIFICATION_PHASES),
        "modes": list(QUALIFICATION_MODES),
        "scientific_environment": {
            "names": list(SCIENTIFIC_ENVIRONMENT_NAMES),
            "fixed_values": dict(SCIENTIFIC_ENVIRONMENT_FIXED_VALUES),
        },
        "qualification_environment": {
            "names": list(QUALIFICATION_ENVIRONMENT_NAMES),
            "fixed_values": dict(QUALIFICATION_ENVIRONMENT),
            "launch_names": list(QUALIFICATION_ENVIRONMENT_NAMES),
            "controlled_pytest_additions": [
                {"name": name, "authority": authority}
                for name, authority
                in QUALIFICATION_CONTROLLED_PYTEST_ENVIRONMENT
            ],
            "pytest_names": list(QUALIFICATION_PYTEST_ENVIRONMENT_NAMES),
        },
        "launcher": {
            "python_flags": list(QUALIFICATION_PYTHON_FLAGS),
            "bootstrap_bytes": QUALIFICATION_BOOTSTRAP_BYTES,
            "bootstrap_sha256": QUALIFICATION_BOOTSTRAP_SHA256,
            "dispatch_prefix": list(QUALIFICATION_DISPATCH_PREFIX),
        },
        "pytest_args": list(QUALIFICATION_PYTEST_ARGS),
        "junit_relative_template": QUALIFICATION_JUNIT_RELATIVE_TEMPLATE,
        "runtime": {
            "python_version": QUALIFICATION_PYTHON_VERSION,
            "python_cache_tag": QUALIFICATION_PYTHON_CACHE_TAG,
            "os_name": QUALIFICATION_OS_NAME,
            "sys_platform": QUALIFICATION_SYS_PLATFORM,
            "executable_basename": QUALIFICATION_EXECUTABLE_BASENAME,
            "executable_bytes": QUALIFICATION_EXECUTABLE_BYTES,
            "executable_sha256": QUALIFICATION_EXECUTABLE_SHA256,
            "pytest_version": QUALIFICATION_PYTEST_VERSION,
            "pytest_init_bytes": QUALIFICATION_PYTEST_INIT_BYTES,
            "pytest_init_sha256": QUALIFICATION_PYTEST_INIT_SHA256,
        },
        "latest": {
            "phase": QUALIFICATION_PHASE_V319,
            "test_paths": list(QUALIFICATION_LATEST_TEST_PATHS),
            "minimum_node_count": QUALIFICATION_LATEST_MIN_NODE_COUNT,
        },
        "dependencies": {
            "phase": QUALIFICATION_PHASE_DEPENDENCIES,
            "test_paths": list(QUALIFICATION_SHARED_TEST_PATHS),
            "node_count": QUALIFICATION_SHARED_NODE_COUNT,
            "case_sensitive_unique_node_count": (
                QUALIFICATION_SHARED_UNIQUE_NODE_COUNT
            ),
            "duplicate_node_count": QUALIFICATION_SHARED_DUPLICATE_NODE_COUNT,
            "multiplicity_policy": QUALIFICATION_SHARED_MULTIPLICITY_POLICY,
            "node_list_sha256": QUALIFICATION_SHARED_NODE_LIST_SHA256,
        },
        "requests_identity": {
            "phase": QUALIFICATION_PHASE_REQUESTS_IDENTITY,
            "node_id": QUALIFICATION_SENTINEL_NODE_ID,
            "node_count": QUALIFICATION_SENTINEL_NODE_COUNT,
            "node_list_sha256": QUALIFICATION_SENTINEL_NODE_LIST_SHA256,
        },
        "local_production_closure": {
            "schema": LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION,
            "derivation": LOCAL_PRODUCTION_CLOSURE_DERIVATION,
            "ordered_roots": list(LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS),
            "explicit_dynamic_local_paths": list(
                LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS
            ),
            "paths": list(LOCAL_PRODUCTION_CLOSURE_PATHS),
            "path_count": len(LOCAL_PRODUCTION_CLOSURE_PATHS),
            "manifest_bytes": LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES,
            "manifest_sha256": LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256,
            "path_array_bytes": LOCAL_PRODUCTION_CLOSURE_PATHS_BYTES,
            "path_array_sha256": LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256,
            "package_bootstrap": LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
            "allowed_import_paths": list(LOCAL_IMPORT_ALLOWED_PATHS),
        },
        "schemas": {
            "collection": QUALIFICATION_COLLECTION_SCHEMA_VERSION,
            "intent": QUALIFICATION_INTENT_SCHEMA_VERSION,
            "result": QUALIFICATION_RESULT_SCHEMA_VERSION,
            "private_aggregate": QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION,
            "public_receipt": QUALIFICATION_PUBLIC_RECEIPT_SCHEMA_VERSION,
        },
        "timeout_seconds": QUALIFICATION_TIMEOUT_SECONDS,
        "durability_mode": QUALIFICATION_DURABILITY_MODE,
    }


def build_effect_budgets() -> dict[str, dict[str, int]]:
    return {
        "zero_effect_preflight": {
            key: (
                EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
                if key == "experiment_family_sec_requests"
                else 0
            )
            for key in EFFECT_COUNT_KEYS
        },
        "normal_complete": _effect_budget(
            identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_DOCUMENT_COUNT,
        ),
        "pilot_pause": _effect_budget(
            identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_PILOT_COUNT,
        ),
        "paused_resumed_complete": _effect_budget(
            identity_requests=PAUSED_RESUMED_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_DOCUMENT_COUNT,
        ),
    }


def validate_command(value: Any) -> str:
    if type(value) is not str or value != DEVELOPMENT_COMMAND:
        _reject("v319_contract_command")
    return DEVELOPMENT_COMMAND


def validate_effect_counts(value: Any, *, route: str) -> dict[str, int]:
    budgets = build_effect_budgets()
    if route not in budgets:
        _reject("v319_contract_effect_route")
    if not isinstance(value, Mapping) or set(value) != set(EFFECT_COUNT_KEYS):
        _reject("v319_contract_effect_keys")
    if any(type(item) is not int or item < 0 for item in value.values()):
        _reject("v319_contract_effect_type")
    observed = dict(value)
    if observed != budgets[route]:
        _reject("v319_contract_effect_counts")
    return copy.deepcopy(observed)


def projected_pilot_ns(durations_ns: Any) -> int:
    """Compute the only allowed five-pilot timing projection."""

    if (
        not isinstance(durations_ns, Sequence)
        or isinstance(durations_ns, (str, bytes, bytearray))
        or len(durations_ns) != DEVELOPMENT_PILOT_COUNT
        or any(type(value) is not int or value <= 0 for value in durations_ns)
    ):
        _reject("v319_contract_pilot_durations")
    return sum(durations_ns) + PILOT_PROJECTION_MULTIPLIER * max(durations_ns)


def pilot_pause_required(durations_ns: Any) -> bool:
    return projected_pilot_ns(durations_ns) > PILOT_PROJECTED_THRESHOLD_NS


_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]+\Z")


def _is_hex(value: Any, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and _HEX_RE.fullmatch(value) is not None
    )


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _reject(code)
    return copy.deepcopy(dict(value))


def _validate_changed_paths(
    value: Any,
    expected: Mapping[str, str],
    *,
    code: str,
) -> None:
    if not isinstance(value, Mapping):
        _reject(code)
    if dict(value) != dict(expected):
        _reject(code)


def validate_local_production_closure_manifest(value: Any) -> dict[str, Any]:
    """Validate the exact 37-path base-blob closure manifest."""

    fields = frozenset(
        {
            "base_commit",
            "base_tree",
            "closure_path_count",
            "derivation",
            "explicit_dynamic_local_paths",
            "ordered_roots",
            "paths",
            "schema",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_closure_shape")
    paths = item["paths"]
    if (
        item["base_commit"] != BASE_COMMIT
        or item["base_tree"] != BASE_TREE
        or item["closure_path_count"] != len(LOCAL_PRODUCTION_CLOSURE_PATHS)
        or item["derivation"] != LOCAL_PRODUCTION_CLOSURE_DERIVATION
        or item["explicit_dynamic_local_paths"]
        != list(LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS)
        or item["ordered_roots"] != list(LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS)
        or item["schema"] != LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION
        or not isinstance(paths, list)
        or len(paths) != len(LOCAL_PRODUCTION_CLOSURE_PATHS)
    ):
        _reject("v319_contract_closure_identity")
    observed_paths: list[str] = []
    for entry in paths:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"git_blob_sha1", "literal_sha256", "path"}
            or not _is_hex(entry.get("git_blob_sha1"), 40)
            or not _is_hex(entry.get("literal_sha256"), 64)
            or type(entry.get("path")) is not str
        ):
            _reject("v319_contract_closure_path_entry")
        observed_paths.append(entry["path"])
    if tuple(observed_paths) != LOCAL_PRODUCTION_CLOSURE_PATHS:
        _reject("v319_contract_closure_paths")
    encoded = canonical_json_bytes(item)
    if (
        len(encoded) != LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES
        or sha256_bytes(encoded) != LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
    ):
        _reject("v319_contract_closure_manifest")
    return item


def validate_local_import_paths(value: Any) -> tuple[str, ...]:
    """Reject any local import outside the preregistered strict upper bound."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(type(path) is not str for path in value)
    ):
        _reject("v319_contract_local_import_shape")
    paths = tuple(value)
    if len(paths) != len(set(paths)):
        _reject("v319_contract_local_import_duplicate")
    allowed = frozenset(LOCAL_IMPORT_ALLOWED_PATHS)
    if any(path not in allowed for path in paths):
        _reject("v319_contract_local_import_outside_closure")
    return paths


def validate_preregistration_ancestry(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "git_blob_sha1",
            "literal_sha256",
            "literal_bytes",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_prereg_shape")
    expected = {
        "branch": BRANCH_NAME,
        "commit": PREREGISTRATION_COMMIT,
        "tree": PREREGISTRATION_TREE,
        "parent": BASE_COMMIT,
        "local_head": PREREGISTRATION_COMMIT,
        "remote_head": PREREGISTRATION_COMMIT,
        "changed_paths": {PREREGISTRATION_PATH: "A"},
        "clean_worktree": True,
        "git_blob_sha1": PREREGISTRATION_GIT_BLOB_SHA1,
        "literal_sha256": PREREGISTRATION_LITERAL_SHA256,
        "literal_bytes": PREREGISTRATION_LITERAL_BYTES,
    }
    if item != expected:
        _reject("v319_contract_prereg_ancestry")
    return item


def validate_implementation_ancestry(value: Any) -> dict[str, Any]:
    """Validate the exact pushed 12-addition implementation commit shape."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "preregistration_authenticated",
            "predecessor_blobs_unchanged",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_impl_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or commit == PREREGISTRATION_COMMIT
        or not _is_hex(item["tree"], 40)
        or item["parent"] != PREREGISTRATION_COMMIT
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["preregistration_authenticated"] is not True
        or item["predecessor_blobs_unchanged"] is not True
    ):
        _reject("v319_contract_impl_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {path: "A" for path in IMPLEMENTATION_ALLOWED_PATHS},
        code="v319_contract_impl_delta",
    )
    return item


def validate_preflight_ancestry(value: Any) -> dict[str, Any]:
    """Validate the pushed one-path, one-run, zero-effect preflight commit."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "implementation_commit",
            "implementation_tree",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "implementation_authenticated",
            "private_replay_passed",
            "zero_effects_verified",
            "preflight_run_count",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_preflight_shape")
    commit = item["commit"]
    implementation_commit = item["implementation_commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(implementation_commit, 40)
        or not _is_hex(item["implementation_tree"], 40)
        or item["parent"] != implementation_commit
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["implementation_authenticated"] is not True
        or item["private_replay_passed"] is not True
        or item["zero_effects_verified"] is not True
        or type(item["preflight_run_count"]) is not int
        or item["preflight_run_count"] != 1
    ):
        _reject("v319_contract_preflight_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {PREFLIGHT_ARTIFACT_PATH: "A"},
        code="v319_contract_preflight_delta",
    )
    return item


def validate_pause_ancestry(value: Any) -> dict[str, Any]:
    """Validate pause commit S as one added artifact above preflight F."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "preflight_commit",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "pilot_guard_authenticated",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_pause_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["preflight_commit"], 40)
        or item["parent"] != item["preflight_commit"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["pilot_guard_authenticated"] is not True
    ):
        _reject("v319_contract_pause_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {PAUSE_ARTIFACT_PATH: "A"},
        code="v319_contract_pause_delta",
    )
    return item


def validate_continuation_ancestry(value: Any) -> dict[str, Any]:
    """Validate continuation commit C as one added document above pause S."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "pause_commit",
            "preflight_commit",
            "local_head",
            "remote_head",
            "changed_paths_from_pause",
            "changed_paths_from_preflight",
            "clean_worktree",
            "earlier_blobs_equal_preflight",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_continue_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["pause_commit"], 40)
        or not _is_hex(item["preflight_commit"], 40)
        or item["parent"] != item["pause_commit"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["earlier_blobs_equal_preflight"] is not True
    ):
        _reject("v319_contract_continue_ancestry")
    _validate_changed_paths(
        item["changed_paths_from_pause"],
        {CONTINUATION_PREREGISTRATION_PATH: "A"},
        code="v319_contract_continue_delta",
    )
    _validate_changed_paths(
        item["changed_paths_from_preflight"],
        {
            PAUSE_ARTIFACT_PATH: "A",
            CONTINUATION_PREREGISTRATION_PATH: "A",
        },
        code="v319_contract_continue_cumulative_delta",
    )
    return item


def validate_result_ancestry(value: Any) -> dict[str, Any]:
    """Validate the exact two-path terminal result commit shape."""

    fields = frozenset(
        {
            "route",
            "branch",
            "commit",
            "tree",
            "parent",
            "authorized_parent",
            "authorized_parent_kind",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "terminal_sealed",
            "independent_replay_passed",
            "privacy_passed",
            "v38_unchanged",
        }
    )
    item = _strict_mapping(value, fields, code="v319_contract_result_shape")
    route = item["route"]
    parent_kind = item["authorized_parent_kind"]
    allowed_parent_kinds = {
        "normal": {"preflight"},
        "paused_resumed": {"continuation"},
        "indeterminate_before_pause": {"preflight", "continuation"},
    }
    commit = item["commit"]
    if (
        route not in allowed_parent_kinds
        or parent_kind not in allowed_parent_kinds[route]
        or item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["authorized_parent"], 40)
        or item["parent"] != item["authorized_parent"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["terminal_sealed"] is not True
        or item["independent_replay_passed"] is not True
        or item["privacy_passed"] is not True
        or item["v38_unchanged"] is not True
    ):
        _reject("v319_contract_result_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {RESULT_ARTIFACT_PATH: "A", COMPARISON_PATH: "M"},
        code="v319_contract_result_delta",
    )
    return item


def _build_unsigned_contract_manifest() -> dict[str, Any]:
    science = build_frozen_science_projection()
    development_gates = copy.deepcopy(science["gates"]["development"])
    if len(development_gates) != 22:
        _reject("v319_contract_development_gate_count")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "branch": BRANCH_NAME,
        "command": DEVELOPMENT_COMMAND,
        "status": "preregistered_implementation_no_external_effect",
        "git_authority": {
            "base_commit": BASE_COMMIT,
            "base_tree": BASE_TREE,
            "base_parent": BASE_PARENT,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_tree": PREREGISTRATION_TREE,
            "preregistration_path": PREREGISTRATION_PATH,
            "preregistration_git_blob_sha1": PREREGISTRATION_GIT_BLOB_SHA1,
            "preregistration_literal_sha256": PREREGISTRATION_LITERAL_SHA256,
            "preregistration_literal_bytes": PREREGISTRATION_LITERAL_BYTES,
        },
        "v316_rejection_authority": build_v316_rejection_pins(),
        "v317_rejection_authority": build_v317_rejection_pins(),
        "v318_rejection_authority": build_v318_rejection_pins(),
        "inherited_v39_preregistration_authority": (
            build_v39_inherited_preregistration_pins()
        ),
        "v310_source_provenance": {
            "implementation_commit": V310_IMPLEMENTATION_COMMIT,
            "implementation_tree": V310_IMPLEMENTATION_TREE,
            "preregistration_commit": V310_PREREGISTRATION_COMMIT,
            "preregistration_path": V310_PREREGISTRATION_PATH,
        },
        "source_authority": build_source_authority_pins(),
        "nullable_source_authority": {
            "source_authority_pins_sha256": SOURCE_AUTHORITY_PINS_SHA256,
            "calendar_session_count": CALENDAR_SESSION_COUNT,
            "calendar_sessions_sha256": CALENDAR_SESSIONS_SHA256,
            "document_count": DEVELOPMENT_DOCUMENT_COUNT,
            "filename_present_count": DEVELOPMENT_FILENAME_PRESENT_COUNT,
            "filename_missing_count": DEVELOPMENT_FILENAME_MISSING_COUNT,
            "schemas": {
                "compatibility_record": NULLABLE_SOURCE_RECORD_SCHEMA_VERSION,
                "private_source_identity": PRIVATE_SOURCE_IDENTITY_SCHEMA_VERSION,
                "compatibility_manifest": COMPATIBILITY_MANIFEST_SCHEMA_VERSION,
                "universe": NULLABLE_UNIVERSE_SCHEMA_VERSION,
                "universe_semantic": NULLABLE_UNIVERSE_SEMANTIC_SCHEMA_VERSION,
                "content": NULLABLE_CONTENT_SCHEMA_VERSION,
                "universe_event_proof": (
                    NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
                ),
                "model_request": BLINDED_MODEL_REQUEST_SCHEMA_VERSION,
                "direction_sanitized_event": (
                    DIRECTION_SANITIZED_EVENT_SCHEMA_VERSION
                ),
                "direction_sanitizer": DIRECTION_SANITIZER_SCHEMA_VERSION,
            },
            "direction_sanitizer_full_corpus_gate": {
                "event_count": DEVELOPMENT_DOCUMENT_COUNT,
                "events_with_removals": 7,
                "removed_sentence_count": 8,
                "empty_current_count": 0,
                "empty_prior_count": 0,
                "contract_failure_count": 0,
                "inherited_production_failure_count": 0,
                "external_effect_count": 0,
            },
            "primary_document_body_kind": PRIMARY_DOCUMENT_BODY_KIND,
            "missing_document_identity": LEGACY_MISSING_DOCUMENT_IDENTITY,
            "prior_selection_policy": PRIOR_SELECTION_POLICY,
            "fabricated_filename_or_url_rows": 0,
        },
        "science_authority": {
            "parent_commit": SCIENTIFIC_PARENT_COMMIT,
            "parent_tree": SCIENTIFIC_PARENT_TREE,
            "contract_path": SCIENTIFIC_CONTRACT_PATH,
            "contract_git_blob_sha1": SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1,
            "contract_literal_sha256": SCIENTIFIC_CONTRACT_LITERAL_SHA256,
            "contract_internal_sha256": SCIENTIFIC_CONTRACT_INTERNAL_SHA256,
            "projection_keys": list(SCIENCE_PROJECTION_KEYS),
            "projection_bytes": SCIENCE_PROJECTION_BYTE_COUNT,
            "projection_sha256": SCIENCE_PROJECTION_SHA256,
            "projection": science,
            "development_gates": development_gates,
            "development_gate_count": 22,
        },
        "development_boundary": {
            "document_count": DEVELOPMENT_DOCUMENT_COUNT,
            "market_start": DEVELOPMENT_MARKET_START,
            "market_end_exclusive": DEVELOPMENT_MARKET_END_EXCLUSIVE,
            "corpus_end": DEVELOPMENT_CORPUS_END,
            "warmup_end": DEVELOPMENT_WARMUP_END,
            "blocks": [list(item) for item in DEVELOPMENT_BLOCKS],
            "source_projection_order": list(SOURCE_PROJECTION_ORDER),
            "science_event_order": list(SCIENCE_EVENT_ORDER),
            "confirmation_and_final_unreachable": True,
        },
        "model": {
            "name": MODEL_NAME,
            "manifest_sha256": MODEL_MANIFEST_SHA256,
            "config_digest": MODEL_CONFIG_DIGEST,
            "ordered_layer_digests": list(MODEL_LAYER_DIGESTS),
            "active_from_blob_count": MODEL_ACTIVE_FROM_BLOB_COUNT,
            "ollama_version": OLLAMA_VERSION,
            "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
            "version_response_sha256": RUNTIME_VERSION_RESPONSE_SHA256,
            "show_semantic_sha256": RUNTIME_SHOW_SEMANTIC_SHA256,
            "show_semantic_excluded_keys": list(
                RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS
            ),
            "show_raw_sha256_diagnostic_only": (
                RUNTIME_SHOW_RAW_SHA256_DIAGNOSTIC_ONLY
            ),
            "model_info_sha256": RUNTIME_MODEL_INFO_SHA256,
            "prompt_sha256": PROMPT_SHA256,
            "schema_sha256": SCHEMA_SHA256,
            "temperature": MODEL_TEMPERATURE,
            "seed": MODEL_SEED,
            "context_tokens": MODEL_CONTEXT_TOKENS,
            "output_tokens": MODEL_OUTPUT_TOKENS,
            "input_caps": {
                "bytes": MODEL_INPUT_MAX_BYTES,
                "canonical_request_bytes": MODEL_REQUEST_MAX_BYTES,
                "sentences": MODEL_INPUT_MAX_SENTENCES,
                "characters_per_sentence": MODEL_INPUT_MAX_SENTENCE_CHARACTERS,
            },
            "response_max_bytes": MODEL_RESPONSE_MAX_BYTES,
            "runtime_probe_order": [list(item) for item in RUNTIME_PROBE_ORDER],
            "chat_endpoint": OLLAMA_CHAT_ENDPOINT,
            "normal_identity_requests": NORMAL_IDENTITY_REQUEST_COUNT,
            "paused_resumed_identity_requests": (
                PAUSED_RESUMED_IDENTITY_REQUEST_COUNT
            ),
            "generation_count": DEVELOPMENT_DOCUMENT_COUNT,
            "retry_repair_pull_fallback_alternate_model": 0,
        },
        "market": {
            "endpoint": YAHOO_ENDPOINT,
            "symbol_order": list(YAHOO_SYMBOL_ORDER),
            "provider_symbols": dict(YAHOO_PROVIDER_SYMBOLS),
            "query_items": [list(item) for item in YAHOO_QUERY_ITEMS],
            "urls": list(YAHOO_URLS),
            "request_count": YAHOO_REQUEST_COUNT,
            "request_timeout_seconds": YAHOO_REQUEST_TIMEOUT_SECONDS,
            "max_response_bytes": YAHOO_MAX_RESPONSE_BYTES,
            "max_total_response_bytes": YAHOO_MAX_TOTAL_RESPONSE_BYTES,
            "max_batch_seconds": YAHOO_MAX_BATCH_SECONDS,
            "user_agent": YAHOO_USER_AGENT,
            "retry_redirect_proxy_cookie_auth_compression_fallback": 0,
        },
        "pilot": {
            "count": DEVELOPMENT_PILOT_COUNT,
            "remaining_count": DEVELOPMENT_REMAINING_COUNT,
            "sort_order": list(PILOT_SORT_ORDER),
            "projection_formula": "sum(durations_ns)+70*max(durations_ns)",
            "projection_multiplier": PILOT_PROJECTION_MULTIPLIER,
            "strict_pause_threshold_ns": PILOT_PROJECTED_THRESHOLD_NS,
            "pause_comparison": "strictly_greater_than",
        },
        "one_shot": {
            "attempt_id": DEVELOPMENT_ATTEMPT_ID,
            "private_namespace": PRIVATE_NAMESPACE,
            "preflight_namespace": PRIVATE_PREFLIGHT_NAMESPACE,
            "development_namespace": PRIVATE_DEVELOPMENT_NAMESPACE,
        },
        "public_paths": {
            "preflight": PREFLIGHT_ARTIFACT_PATH,
            "pause": PAUSE_ARTIFACT_PATH,
            "continuation": CONTINUATION_PREREGISTRATION_PATH,
            "result": RESULT_ARTIFACT_PATH,
            "comparison": COMPARISON_PATH,
        },
        "implementation": {
            "production_paths": list(IMPLEMENTATION_PRODUCTION_PATHS),
            "test_paths": list(IMPLEMENTATION_TEST_PATHS),
            "all_paths": list(IMPLEMENTATION_ALLOWED_PATHS),
            "all_changes_are_additions": True,
            "predecessor_blobs_unchanged": True,
            "package_bootstrap_path": LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
            "local_import_allowed_paths": list(LOCAL_IMPORT_ALLOWED_PATHS),
            "unresolved_or_outside_local_import_terminal": True,
        },
        "qualification": build_qualification_contract(),
        "effects": {
            "count_keys": list(EFFECT_COUNT_KEYS),
            "budgets": build_effect_budgets(),
            "high_level_order": list(HIGH_LEVEL_EFFECT_ORDER),
        },
        "privacy": {
            "readable_sec_contact_public": False,
            "accessions_public": False,
            "source_or_response_bodies_public": False,
            "canonical_model_requests_public": False,
            "realized_response_metadata_public": False,
            "pre_release_science_values_public": False,
            "redacted_errors_only": True,
        },
    }


# This literal is filled from the exact unsigned manifest above.  Its import-time
# assertion makes accidental edits a hard failure rather than a new contract.
CONTRACT_MANIFEST_SHA256: Final[str] = (
    "fd77730b9483ccdff930a8b398aa71a4a3fabc77fee9a26dcb6c8cb50b9331d7"
)


def build_contract_manifest() -> dict[str, Any]:
    unsigned = _build_unsigned_contract_manifest()
    observed = canonical_sha256(unsigned)
    if observed != CONTRACT_MANIFEST_SHA256:
        _reject("v319_contract_manifest_literal")
    return {**unsigned, "contract_manifest_sha256": observed}


def validate_contract_manifest(value: Any) -> dict[str, Any]:
    """Accept only the exact self-hashed v3.19 manifest."""

    if not isinstance(value, Mapping):
        _reject("v319_contract_manifest_shape")
    candidate = validate_self_sha256(value, field="contract_manifest_sha256")
    expected = build_contract_manifest()
    if canonical_json_bytes(candidate) != canonical_json_bytes(expected):
        _reject("v319_contract_manifest_mismatch")
    return copy.deepcopy(expected)


__all__ = [
    "BASE_COMMIT",
    "BASE_PARENT",
    "BASE_TREE",
    "BRANCH_NAME",
    "COMPARISON_PATH",
    "CONTRACT_MANIFEST_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "CONTINUATION_PREREGISTRATION_PATH",
    "ContractViolation",
    "DEVELOPMENT_ATTEMPT_ID",
    "DEVELOPMENT_BLOCKS",
    "DEVELOPMENT_COMMAND",
    "DEVELOPMENT_DOCUMENT_COUNT",
    "DEVELOPMENT_PILOT_COUNT",
    "DEVELOPMENT_REMAINING_COUNT",
    "EFFECT_COUNT_KEYS",
    "EXPERIMENT_FAMILY_SEC_REQUEST_COUNT",
    "FROZEN_SCIENCE_PROJECTION_SHA256",
    "HIGH_LEVEL_EFFECT_ORDER",
    "IMPLEMENTATION_ALLOWED_PATHS",
    "IMPLEMENTATION_PRODUCTION_PATHS",
    "IMPLEMENTATION_TEST_PATHS",
    "MODEL_ACTIVE_FROM_BLOB_COUNT",
    "MODEL_CONFIG_DIGEST",
    "MODEL_CONTEXT_TOKENS",
    "MODEL_INPUT_MAX_BYTES",
    "MODEL_REQUEST_MAX_BYTES",
    "MODEL_INPUT_MAX_SENTENCES",
    "MODEL_INPUT_MAX_SENTENCE_CHARACTERS",
    "MODEL_LAYER_DIGESTS",
    "MODEL_MANIFEST_SHA256",
    "MODEL_NAME",
    "MODEL_OUTPUT_TOKENS",
    "MODEL_RESPONSE_MAX_BYTES",
    "MODEL_SEED",
    "MODEL_TEMPERATURE",
    "NORMAL_EFFECT_BUDGET",
    "NORMAL_IDENTITY_REQUEST_COUNT",
    "OLLAMA_CHAT_ENDPOINT",
    "OLLAMA_SHOW_ENDPOINT",
    "OLLAMA_VERSION",
    "OLLAMA_VERSION_ENDPOINT",
    "PAUSED_RESUMED_EFFECT_BUDGET",
    "PAUSED_RESUMED_IDENTITY_REQUEST_COUNT",
    "PAUSE_ARTIFACT_PATH",
    "PILOT_PAUSE_EFFECT_BUDGET",
    "PILOT_PROJECTED_THRESHOLD_NS",
    "PREFLIGHT_ARTIFACT_PATH",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_GIT_BLOB_SHA1",
    "PREREGISTRATION_LITERAL_BYTES",
    "PREREGISTRATION_LITERAL_SHA256",
    "PREREGISTRATION_PATH",
    "PREREGISTRATION_TREE",
    "PRIVATE_DEVELOPMENT_NAMESPACE",
    "PRIVATE_NAMESPACE",
    "PRIVATE_PREFLIGHT_NAMESPACE",
    "PROMPT_SHA256",
    "RESULT_ARTIFACT_PATH",
    "RUNTIME_FINGERPRINT_SHA256",
    "RUNTIME_MODEL_INFO_SHA256",
    "RUNTIME_PROBE_ORDER",
    "RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS",
    "RUNTIME_SHOW_SEMANTIC_SHA256",
    "RUNTIME_VERSION_RESPONSE_SHA256",
    "SCHEMA_SHA256",
    "SCIENCE_EVENT_ORDER",
    "SCIENCE_PROJECTION_BYTE_COUNT",
    "SCIENCE_PROJECTION_KEYS",
    "SCIENCE_PROJECTION_SHA256",
    "SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1",
    "SCIENTIFIC_CONTRACT_INTERNAL_SHA256",
    "SCIENTIFIC_CONTRACT_LITERAL_SHA256",
    "SCIENTIFIC_CONTRACT_PATH",
    "SCIENTIFIC_PARENT_COMMIT",
    "SCIENTIFIC_PARENT_TREE",
    "SOURCE_PROJECTION_ORDER",
    "V38_ACQUISITION_MODULE_GIT_BLOB_SHA1",
    "V38_ACQUISITION_MODULE_LITERAL_SHA256",
    "V38_ACQUISITION_MODULE_PATH",
    "V38_CHECKPOINT_FILE_SHA256",
    "V38_COMPACT_REPLAY_SHA256",
    "V38_INVENTORY_BYTE_COUNT",
    "V38_INVENTORY_FILE_COUNT",
    "V38_INVENTORY_SHA256",
    "V38_LOGICAL_CHECKPOINT_SHA256",
    "V38_ROLE_COUNTS",
    "V38_ROLE_MANIFEST_INVENTORY_SHA256",
    "V38_ROLE_PLAN_SHA256",
    "V38_SEC_REQUEST_COUNT",
    "V38_SOURCE_AUTHORITY_BLOB_SHA1",
    "V38_SOURCE_AUTHORITY_LITERAL_SHA256",
    "V38_SOURCE_AUTHORITY_PATH",
    "V38_SOURCE_MODULE_GIT_BLOB_SHA1",
    "V38_SOURCE_MODULE_LITERAL_SHA256",
    "V38_SOURCE_MODULE_PATH",
    "V38_STAGE_SOURCE_SEAL_SHA256",
    "V38_TERMINAL_BLOB_SHA1",
    "V38_TERMINAL_INTERNAL_SHA256",
    "V38_TERMINAL_LITERAL_SHA256",
    "V38_TERMINAL_PATH",
    "YAHOO_ENDPOINT",
    "YAHOO_MAX_BATCH_SECONDS",
    "YAHOO_MAX_RESPONSE_BYTES",
    "YAHOO_MAX_TOTAL_RESPONSE_BYTES",
    "YAHOO_PROVIDER_SYMBOLS",
    "YAHOO_QUERY_ITEMS",
    "YAHOO_REQUEST_COUNT",
    "YAHOO_REQUEST_TIMEOUT_SECONDS",
    "YAHOO_SYMBOL_ORDER",
    "YAHOO_URLS",
    "YAHOO_USER_AGENT",
    "ZERO_EFFECT_BUDGET",
    "add_self_sha256",
    "build_contract_manifest",
    "build_effect_budgets",
    "build_frozen_science_projection",
    "build_source_authority_pins",
    "canonical_json_bytes",
    "canonical_sha256",
    "pilot_pause_required",
    "project_science_manifest",
    "projected_pilot_ns",
    "sha256_bytes",
    "validate_command",
    "validate_continuation_ancestry",
    "validate_contract_manifest",
    "validate_effect_counts",
    "validate_implementation_ancestry",
    "validate_pause_ancestry",
    "validate_preflight_ancestry",
    "validate_preregistration_ancestry",
    "validate_result_ancestry",
    "validate_self_sha256",
    "verify_frozen_science_projection",
]

__all__.extend(
    [
        "LOCAL_IMPORT_ALLOWED_PATHS",
        "LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH",
        "LOCAL_PRODUCTION_CLOSURE_DERIVATION",
        "LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS",
        "LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES",
        "LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256",
        "LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS",
        "LOCAL_PRODUCTION_CLOSURE_PATHS",
        "LOCAL_PRODUCTION_CLOSURE_PATHS_BYTES",
        "LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256",
        "LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION",
        "SCIENTIFIC_ENVIRONMENT_FIXED_VALUES",
        "SCIENTIFIC_ENVIRONMENT_NAMES",
        "QUALIFICATION_BOOTSTRAP_BYTES",
        "QUALIFICATION_BOOTSTRAP_LITERAL",
        "QUALIFICATION_BOOTSTRAP_SHA256",
        "QUALIFICATION_CANONICAL_ROOT_TOKEN",
        "QUALIFICATION_COLLECTION_SCHEMA_VERSION",
        "QUALIFICATION_DISPATCH_PREFIX",
        "QUALIFICATION_DURABILITY_MODE",
        "QUALIFICATION_CONTROLLED_PYTEST_ENVIRONMENT",
        "QUALIFICATION_ENVIRONMENT",
        "QUALIFICATION_ENVIRONMENT_NAMES",
        "QUALIFICATION_EXECUTABLE_BASENAME",
        "QUALIFICATION_EXECUTABLE_BYTES",
        "QUALIFICATION_EXECUTABLE_SHA256",
        "QUALIFICATION_INTENT_SCHEMA_VERSION",
        "QUALIFICATION_JUNIT_RELATIVE_TEMPLATE",
        "QUALIFICATION_LATEST_MIN_NODE_COUNT",
        "QUALIFICATION_LATEST_TEST_PATHS",
        "QUALIFICATION_MODES",
        "QUALIFICATION_OS_NAME",
        "QUALIFICATION_PHASES",
        "QUALIFICATION_PHASE_DEPENDENCIES",
        "QUALIFICATION_PHASE_REQUESTS_IDENTITY",
        "QUALIFICATION_PHASE_V319",
        "QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION",
        "QUALIFICATION_PUBLIC_RECEIPT_SCHEMA_VERSION",
        "QUALIFICATION_PYTHON_FLAGS",
        "QUALIFICATION_PYTEST_ARGS",
        "QUALIFICATION_PYTEST_ENVIRONMENT_NAMES",
        "QUALIFICATION_PYTEST_INIT_BYTES",
        "QUALIFICATION_PYTEST_INIT_SHA256",
        "QUALIFICATION_PYTEST_VERSION",
        "QUALIFICATION_PYTHON_CACHE_TAG",
        "QUALIFICATION_PYTHON_VERSION",
        "QUALIFICATION_RESULT_SCHEMA_VERSION",
        "QUALIFICATION_SENTINEL_NODE_COUNT",
        "QUALIFICATION_SENTINEL_NODE_ID",
        "QUALIFICATION_SENTINEL_NODE_LIST_SHA256",
        "QUALIFICATION_SHARED_NODE_COUNT",
        "QUALIFICATION_SHARED_DUPLICATE_NODE_COUNT",
        "QUALIFICATION_SHARED_NODE_LIST_SHA256",
        "QUALIFICATION_SHARED_MULTIPLICITY_POLICY",
        "QUALIFICATION_SHARED_TEST_PATHS",
        "QUALIFICATION_SHARED_UNIQUE_NODE_COUNT",
        "QUALIFICATION_SUITE_ID",
        "QUALIFICATION_SYS_PLATFORM",
        "QUALIFICATION_TIMEOUT_SECONDS",
        "V38_SOURCE_COMMIT",
        "V38_SOURCE_PARENT",
        "V38_SOURCE_TREE",
        "V39_INHERITED_PREREG_COMMIT",
        "V39_INHERITED_PREREG_GIT_BLOB_SHA1",
        "V39_INHERITED_PREREG_LITERAL_BYTES",
        "V39_INHERITED_PREREG_LITERAL_SHA256",
        "V39_INHERITED_PREREG_PARENT",
        "V39_INHERITED_PREREG_PATH",
        "V39_INHERITED_PREREG_TREE",
        "build_qualification_contract",
        "build_v316_rejection_pins",
        "build_v317_rejection_pins",
        "build_v318_rejection_pins",
        "build_v39_inherited_preregistration_pins",
        "validate_local_import_paths",
        "validate_local_production_closure_manifest",
    ]
)

__all__.extend(
    [
        "BLINDED_MODEL_REQUEST_FIELDS",
        "BLINDED_MODEL_REQUEST_SCHEMA_VERSION",
        "DIRECTION_SANITIZED_EVENT_FIELDS",
        "DIRECTION_SANITIZED_EVENT_SCHEMA_VERSION",
        "DIRECTION_SANITIZER_SCHEMA_VERSION",
        "CALENDAR_SESSION_COUNT",
        "CALENDAR_SESSIONS_SHA256",
        "COMPATIBILITY_MANIFEST_SCHEMA_VERSION",
        "COMPATIBILITY_RECORD_FIELDS",
        "DEVELOPMENT_FILENAME_MISSING_COUNT",
        "DEVELOPMENT_FILENAME_PRESENT_COUNT",
        "LEGACY_MISSING_DOCUMENT_IDENTITY",
        "NULLABLE_CONTENT_RECORD_FIELDS",
        "NULLABLE_CONTENT_SCHEMA_VERSION",
        "NULLABLE_SOURCE_RECORD_SCHEMA_VERSION",
        "NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION",
        "NULLABLE_UNIVERSE_RECORD_FIELDS",
        "NULLABLE_UNIVERSE_SCHEMA_VERSION",
        "NULLABLE_UNIVERSE_SEMANTIC_SCHEMA_VERSION",
        "PRIMARY_DOCUMENT_BODY_KIND",
        "PRIVATE_SOURCE_IDENTITY_FIELDS",
        "PRIVATE_SOURCE_IDENTITY_SCHEMA_VERSION",
        "PRIOR_SELECTION_POLICY",
        "SOURCE_AUTHORITY_PINS_SHA256",
        "UNIVERSE_EVENT_PROOF_FIELDS",
        "V310_IMPLEMENTATION_COMMIT",
        "V310_IMPLEMENTATION_TREE",
        "V310_PREREGISTRATION_COMMIT",
        "V310_PREREGISTRATION_PATH",
        "build_nullable_model_slice",
        "build_direction_sanitized_preprocessed_event",
        "strip_exact_sha256_tag",
        "validate_blinded_model_request",
        "validate_direction_sanitized_preprocessed_event",
        "validate_compatibility_manifest",
        "validate_compatibility_record",
        "validate_nullable_content_manifest",
        "validate_nullable_content_record",
        "validate_nullable_model_slice",
        "validate_nullable_universe_event_proof",
        "validate_nullable_universe_manifest",
        "validate_nullable_universe_record",
    ]
)
