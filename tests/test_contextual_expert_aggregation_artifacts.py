from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.contextual_expert_aggregation import GLOBAL_ONLY_MODE
from agent_benchmark.contextual_expert_aggregation_artifacts import (
    ARM_ORDER,
    CHECKPOINT_CUTOFF_BY_STAGE,
    CONFIRMATION_PAYLOAD_NAMES,
    CONFIRMATION_STAGE,
    COST_BPS,
    COST_ORDER,
    CanonicalTableSchema,
    ContextualExpertAggregationArtifactError,
    DEVELOPMENT_PAYLOAD_NAMES,
    DEVELOPMENT_STAGE,
    FIXED_POLICY_ORDER,
    FIRST_CONFIRMATION_SESSION,
    GIT_ATTRIBUTES_BYTES,
    LAST_OBSERVED_SESSION_BY_STAGE,
    OUTPUT_DIRECTORY_BY_STAGE,
    POLICY_ORDER,
    RUN_ID_BY_STAGE,
    SOURCE_SESSION_COUNT_BY_STAGE,
    SOURCE_START_SESSION_BY_STAGE,
    build_arm_seed_equivalence,
    build_composite_stage_checkpoint,
    canonical_table_bytes,
    composite_stage_checkpoint_bytes,
    parse_canonical_table_bytes,
    parse_composite_stage_checkpoint,
    parse_composite_stage_checkpoint_bytes,
    payload_names_for_stage,
)
from agent_benchmark.contextual_expert_aggregation_ledger import (
    run_continuous_ledger,
)
from agent_benchmark.contextual_expert_aggregation_replay import (
    GLOBAL_ONLY_ARM,
    ONLINE_FULL_ARM,
    continue_from_checkpoint,
    fork_confirmation_arms,
    replay_from_empty,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _rehash_checkpoint(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result.pop("checkpoint_sha256", None)
    result["checkpoint_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(result)
    ).hexdigest()
    return result


def _market_frame(
    *, end: str, periods: int = 130, first_session: str | None = None
) -> pd.DataFrame:
    if first_session is None:
        index = pd.bdate_range(end=end, periods=periods, name="date")
    else:
        tail = pd.date_range(end=end, periods=periods - 1, freq="D")
        first = pd.Timestamp(first_session)
        if first >= tail[0]:
            raise ValueError("synthetic first session must precede its tail")
        index = pd.DatetimeIndex([first, *tail], name="date")
    values = np.linspace(100.0, 110.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values + 20.0,
            "qqq_adj_close": values + 30.0,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def development_market() -> pd.DataFrame:
    return _market_frame(
        end="2018-12-31",
        periods=SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE],
        first_session=SOURCE_START_SESSION_BY_STAGE[DEVELOPMENT_STAGE],
    )


@pytest.fixture(scope="module")
def development_replay(development_market: pd.DataFrame):
    return replay_from_empty(development_market)


@pytest.fixture(scope="module")
def confirmation_forks(development_market: pd.DataFrame, development_replay):
    suffix = _market_frame(
        end="2023-12-29",
        periods=(
            SOURCE_SESSION_COUNT_BY_STAGE[CONFIRMATION_STAGE]
            - SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE]
        ),
        first_session=FIRST_CONFIRMATION_SESSION,
    )
    return fork_confirmation_arms(
        suffix,
        development_replay.checkpoint,
        historical_prefix=development_market,
    )


def _accounts(
    stage: str,
    *,
    changed_arm: tuple[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    last_observed = LAST_OBSERVED_SESSION_BY_STAGE[stage]
    dates = pd.DatetimeIndex(["2005-01-03", last_observed], name="date")
    opens = pd.Series([100.0, 120.0], index=dates, dtype=float)
    result: dict[str, dict[str, object]] = {}
    for cost in COST_ORDER:
        result[cost] = {}
        for policy in POLICY_ORDER:
            targets = [1, 1]
            if changed_arm == (cost, policy):
                targets = [1, 0]
            run = run_continuous_ledger(
                opens,
                pd.Series(targets, index=dates, dtype=int),
                policy_name=policy,
                cost_bps=COST_BPS[cost],
            )
            result[cost][policy] = run.state.to_checkpoint()
    return result


def _development_checkpoint(development_replay) -> dict[str, object]:
    return build_composite_stage_checkpoint(
        stage=DEVELOPMENT_STAGE,
        replay_checkpoints={arm: development_replay.checkpoint for arm in ARM_ORDER},
        administrative_accounts=_accounts(DEVELOPMENT_STAGE),
    )


def _confirmation_checkpoint(confirmation_forks) -> dict[str, object]:
    return build_composite_stage_checkpoint(
        stage=CONFIRMATION_STAGE,
        replay_checkpoints={
            arm: confirmation_forks.arms[arm].checkpoint for arm in ARM_ORDER
        },
        administrative_accounts=_accounts(CONFIRMATION_STAGE),
    )


def test_orders_run_ids_paths_and_payload_inventories_are_frozen() -> None:
    assert ARM_ORDER == (
        "online_full",
        "frozen_2018",
        "global_only",
        "lifetime_only",
    )
    assert FIXED_POLICY_ORDER == (
        "always_long",
        "exact_union_cash",
        "contextual_only",
        "weak_trend_only",
    )
    assert POLICY_ORDER == (*ARM_ORDER, *FIXED_POLICY_ORDER, "aapl_buy_hold")
    assert COST_ORDER == ("base_5bps", "stress_10bps")
    assert GIT_ATTRIBUTES_BYTES == b"* -text\n"
    assert SOURCE_START_SESSION_BY_STAGE == {
        DEVELOPMENT_STAGE: "1999-03-10",
        CONFIRMATION_STAGE: "1999-03-10",
    }
    assert SOURCE_SESSION_COUNT_BY_STAGE == {
        DEVELOPMENT_STAGE: 4_986,
        CONFIRMATION_STAGE: 6_244,
    }
    assert FIRST_CONFIRMATION_SESSION == "2019-01-02"
    assert RUN_ID_BY_STAGE == {
        "development": "contextual-expert-aggregation-development-v1",
        "confirmation": "contextual-expert-aggregation-confirmation-v1",
    }
    assert OUTPUT_DIRECTORY_BY_STAGE[DEVELOPMENT_STAGE] == Path(
        "e/aapl_causal_contextual_expert_aggregation_v1/"
        "contextual-expert-aggregation-development-v1"
    )
    assert len(DEVELOPMENT_PAYLOAD_NAMES) == 47
    assert len(CONFIRMATION_PAYLOAD_NAMES) == 50
    assert payload_names_for_stage(DEVELOPMENT_STAGE) is DEVELOPMENT_PAYLOAD_NAMES
    assert payload_names_for_stage(CONFIRMATION_STAGE) is CONFIRMATION_PAYLOAD_NAMES
    assert "development_arm_seed_equivalence.json" in DEVELOPMENT_PAYLOAD_NAMES
    assert "confirmation_attempt_authorization.json" in CONFIRMATION_PAYLOAD_NAMES
    assert "development_parent_manifest.json" in CONFIRMATION_PAYLOAD_NAMES
    expected_ledgers = {
        f"development_ledger__{cost}__{policy}.table.json"
        for cost in COST_ORDER
        for policy in POLICY_ORDER
    }
    assert expected_ledgers.issubset(DEVELOPMENT_PAYLOAD_NAMES)
    assert not expected_ledgers.intersection(CONFIRMATION_PAYLOAD_NAMES)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        payload_names_for_stage("audit")


@pytest.fixture()
def table_schema() -> CanonicalTableSchema:
    return CanonicalTableSchema(
        columns=(
            "flag",
            "count",
            "score",
            "label",
            "event_date",
            "optional_flag",
            "optional_count",
            "optional_score",
            "optional_label",
            "optional_date",
        ),
        column_types=(
            "bool",
            "int",
            "float",
            "string",
            "iso_date",
            "nullable_bool",
            "nullable_int",
            "nullable_float",
            "nullable_string",
            "nullable_iso_date",
        ),
        index_name="date",
        index_type="iso_date",
    )


@pytest.fixture()
def table_frame(table_schema: CanonicalTableSchema) -> pd.DataFrame:
    rows = [
        [True, 1, 1.25, "alpha", "2020-01-02", None, None, None, None, None],
        [
            False,
            2,
            -0.0,
            "beta",
            "2020-01-03",
            True,
            3,
            2.5,
            "x",
            "2020-01-04",
        ],
    ]
    frame = pd.DataFrame(rows, columns=table_schema.columns, dtype=object)
    frame.index = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03"], name="date"
    )
    return frame


def test_canonical_table_round_trip_preserves_explicit_primitives(
    table_schema: CanonicalTableSchema, table_frame: pd.DataFrame
) -> None:
    payload = canonical_table_bytes(table_frame, schema=table_schema)
    parsed = parse_canonical_table_bytes(payload, schema=table_schema)
    assert canonical_table_bytes(parsed, schema=table_schema) == payload
    assert type(parsed.iloc[0]["flag"]) is bool
    assert type(parsed.iloc[0]["count"]) is int
    assert type(parsed.iloc[0]["score"]) is float
    assert parsed.iloc[0]["optional_score"] is None
    assert math.copysign(1.0, parsed.iloc[1]["score"]) == -1.0


def test_canonical_unindexed_table_requires_exact_default_range_index() -> None:
    schema = CanonicalTableSchema(
        columns=("name", "value"), column_types=("string", "float")
    )
    frame = pd.DataFrame([["x", 1.0]], columns=schema.columns, dtype=object)
    payload = canonical_table_bytes(frame, schema=schema)
    assert canonical_table_bytes(
        parse_canonical_table_bytes(payload, schema=schema), schema=schema
    ) == payload
    changed = frame.copy()
    changed.index = pd.Index([1])
    with pytest.raises(ContextualExpertAggregationArtifactError):
        canonical_table_bytes(changed, schema=schema)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("flag", 1),
        ("count", True),
        ("count", 1.0),
        ("score", 1),
        ("score", "1.25"),
        ("score", float("nan")),
        ("score", float("inf")),
        ("label", 4),
        ("event_date", "2020-1-02"),
        ("event_date", None),
        ("optional_count", "3"),
        ("optional_score", float("nan")),
    ],
)
def test_canonical_table_rejects_bool_int_float_string_date_and_null_confusion(
    table_schema: CanonicalTableSchema,
    table_frame: pd.DataFrame,
    column: str,
    bad_value: object,
) -> None:
    changed = table_frame.copy()
    changed.at[changed.index[0], column] = bad_value
    with pytest.raises(ContextualExpertAggregationArtifactError):
        canonical_table_bytes(changed, schema=table_schema)


def test_canonical_table_rejects_noncanonical_index_date(
    table_schema: CanonicalTableSchema, table_frame: pd.DataFrame
) -> None:
    changed = table_frame.copy()
    changed.index = pd.Index(["2020-1-02", "2020-01-03"], name="date")
    with pytest.raises(ContextualExpertAggregationArtifactError):
        canonical_table_bytes(changed, schema=table_schema)


@pytest.mark.parametrize("mutation", ["extra", "missing", "columns", "types", "row"])
def test_table_parser_rejects_missing_extra_reordered_or_misshaped_envelope(
    table_schema: CanonicalTableSchema,
    table_frame: pd.DataFrame,
    mutation: str,
) -> None:
    value = json.loads(canonical_table_bytes(table_frame, schema=table_schema))
    if mutation == "extra":
        value["extra"] = False
    elif mutation == "missing":
        value.pop("index_type")
    elif mutation == "columns":
        value["columns"] = list(reversed(value["columns"]))
    elif mutation == "types":
        value["column_types"][0:2] = ["int", "bool"]
    else:
        value["rows"][0].pop()
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_canonical_table_bytes(_canonical_bytes(value), schema=table_schema)


def test_table_parser_rejects_noncanonical_json_and_primitive_tamper(
    table_schema: CanonicalTableSchema, table_frame: pd.DataFrame
) -> None:
    payload = canonical_table_bytes(table_frame, schema=table_schema)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_canonical_table_bytes(b" " + payload, schema=table_schema)

    value = json.loads(payload)
    value["rows"][0][1] = True
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_canonical_table_bytes(_canonical_bytes(value), schema=table_schema)

    value = json.loads(payload)
    value["rows"][0][2] = None
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_canonical_table_bytes(_canonical_bytes(value), schema=table_schema)


def test_development_composite_checkpoint_round_trip_and_seed_proof(
    development_replay,
) -> None:
    payload = _development_checkpoint(development_replay)
    parsed = parse_composite_stage_checkpoint(payload)
    assert parsed.stage == DEVELOPMENT_STAGE
    assert parsed.checkpoint_cutoff == "2018-12-31"
    assert parsed.arm_seed_equivalence is not None
    assert parsed.to_dict() == payload
    assert parse_composite_stage_checkpoint_bytes(
        composite_stage_checkpoint_bytes(payload)
    ).to_dict() == payload
    proof = build_arm_seed_equivalence(_accounts(DEVELOPMENT_STAGE))
    assert proof == payload["arm_seed_equivalence"]
    assert all(
        proof["by_cost"][cost]["all_equal"] is True for cost in COST_ORDER
    )


def test_confirmation_composite_checkpoint_round_trip_and_runtime_modes(
    confirmation_forks,
) -> None:
    payload = _confirmation_checkpoint(confirmation_forks)
    parsed = parse_composite_stage_checkpoint(payload)
    assert parsed.stage == CONFIRMATION_STAGE
    assert parsed.checkpoint_cutoff == "2023-12-31"
    assert parsed.last_observed_session == "2023-12-29"
    assert parsed.arm_seed_equivalence is None
    assert (
        parsed.replay_checkpoints[GLOBAL_ONLY_ARM].model_payload["runtime"][
            "ablation_mode"
        ]
        == "global_only"
    )


def test_confirmation_checkpoint_rejects_same_bound_different_market_arm(
    development_market: pd.DataFrame,
    development_replay,
    confirmation_forks,
) -> None:
    alternate_suffix = _market_frame(
        end="2023-12-29",
        periods=(
            SOURCE_SESSION_COUNT_BY_STAGE[CONFIRMATION_STAGE]
            - SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE]
        ),
        first_session=FIRST_CONFIRMATION_SESSION,
    )
    for column in alternate_suffix.columns:
        alternate_suffix[column] = alternate_suffix[column] + 17.0
    alternate_forks = fork_confirmation_arms(
        alternate_suffix,
        development_replay.checkpoint,
        historical_prefix=development_market,
    )
    mixed = {
        arm: confirmation_forks.arms[arm].checkpoint for arm in ARM_ORDER
    }
    mixed["frozen_2018"] = alternate_forks.arms["frozen_2018"].checkpoint

    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="action-independent source identity",
    ):
        build_composite_stage_checkpoint(
            stage=CONFIRMATION_STAGE,
            replay_checkpoints=mixed,
            administrative_accounts=_accounts(CONFIRMATION_STAGE),
        )


def test_confirmation_checkpoint_rejects_valid_early_fork() -> None:
    early_prefix = _market_frame(
        end="2017-12-29",
        periods=SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE],
        first_session=SOURCE_START_SESSION_BY_STAGE[DEVELOPMENT_STAGE],
    )
    early = replay_from_empty(early_prefix)
    suffix = _market_frame(
        end="2023-12-29",
        periods=(
            SOURCE_SESSION_COUNT_BY_STAGE[CONFIRMATION_STAGE]
            - SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE]
        ),
        first_session=FIRST_CONFIRMATION_SESSION,
    )
    checkpoints = {
        ONLINE_FULL_ARM: continue_from_checkpoint(
            suffix,
            early.checkpoint,
            historical_prefix=early_prefix,
        ).checkpoint,
        "frozen_2018": continue_from_checkpoint(
            suffix,
            early.checkpoint,
            historical_prefix=early_prefix,
            learning_mode="frozen_cutoff",
            frozen_cutoff="2018-12-31",
        ).checkpoint,
        "global_only": continue_from_checkpoint(
            suffix,
            early.checkpoint,
            historical_prefix=early_prefix,
            ablation_mode="global_only",
        ).checkpoint,
        "lifetime_only": continue_from_checkpoint(
            suffix,
            early.checkpoint,
            historical_prefix=early_prefix,
            ablation_mode="lifetime_only",
        ).checkpoint,
    }

    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="exact development cutoff",
    ):
        build_composite_stage_checkpoint(
            stage=CONFIRMATION_STAGE,
            replay_checkpoints=checkpoints,
            administrative_accounts=_accounts(CONFIRMATION_STAGE),
        )


@pytest.mark.parametrize(
    "mutation",
    ["extra", "missing", "arm_order", "policy_order", "cost_order", "cutoff"],
)
def test_composite_checkpoint_rejects_missing_extra_reordered_and_cutoff_fields(
    development_replay,
    mutation: str,
) -> None:
    value = copy.deepcopy(_development_checkpoint(development_replay))
    if mutation == "extra":
        value["extra"] = 1
    elif mutation == "missing":
        value.pop("last_observed_session")
    elif mutation == "arm_order":
        value["arm_order"] = list(reversed(value["arm_order"]))
        value = _rehash_checkpoint(value)
    elif mutation == "policy_order":
        value["policy_order"][0:2] = list(reversed(value["policy_order"][0:2]))
        value = _rehash_checkpoint(value)
    elif mutation == "cost_order":
        value["cost_order"] = list(reversed(value["cost_order"]))
        value = _rehash_checkpoint(value)
    else:
        value["checkpoint_cutoff"] = "2018-12-30"
        value = _rehash_checkpoint(value)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(value)


def test_composite_checkpoint_rejects_hash_tamper_and_noncanonical_bytes(
    development_replay,
) -> None:
    value = _development_checkpoint(development_replay)
    value["checkpoint_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(value)

    canonical = composite_stage_checkpoint_bytes(
        _development_checkpoint(development_replay)
    )
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint_bytes(canonical + b"\n")


def test_composite_checkpoint_rejects_valid_but_wrong_terminal_runtime(
    development_market: pd.DataFrame,
) -> None:
    global_replay = replay_from_empty(
        development_market, ablation_mode=GLOBAL_ONLY_MODE
    )
    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="terminal runtime",
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={arm: global_replay.checkpoint for arm in ARM_ORDER},
            administrative_accounts=_accounts(DEVELOPMENT_STAGE),
        )


def test_composite_checkpoint_rejects_policy_and_cost_mismatch(
    development_replay,
) -> None:
    accounts = _accounts(DEVELOPMENT_STAGE)
    accounts["base_5bps"][ONLINE_FULL_ARM] = accounts["base_5bps"][
        GLOBAL_ONLY_ARM
    ]
    with pytest.raises(
        ContextualExpertAggregationArtifactError, match="policy, cost"
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={arm: development_replay.checkpoint for arm in ARM_ORDER},
            administrative_accounts=accounts,
        )

    accounts = _accounts(DEVELOPMENT_STAGE)
    accounts["base_5bps"][ONLINE_FULL_ARM] = accounts["stress_10bps"][
        ONLINE_FULL_ARM
    ]
    with pytest.raises(
        ContextualExpertAggregationArtifactError, match="policy, cost"
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={arm: development_replay.checkpoint for arm in ARM_ORDER},
            administrative_accounts=accounts,
        )


def test_development_seed_equivalence_rejects_economic_tamper(
    development_replay,
) -> None:
    accounts = _accounts(
        DEVELOPMENT_STAGE, changed_arm=("stress_10bps", GLOBAL_ONLY_ARM)
    )
    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="not identical online-target seeds",
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={arm: development_replay.checkpoint for arm in ARM_ORDER},
            administrative_accounts=accounts,
        )


def test_always_long_and_buy_hold_account_seed_mismatch_is_rejected(
    development_replay,
) -> None:
    accounts = _accounts(
        DEVELOPMENT_STAGE, changed_arm=("base_5bps", "aapl_buy_hold")
    )
    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="always-LONG and buy-and-hold",
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={arm: development_replay.checkpoint for arm in ARM_ORDER},
            administrative_accounts=accounts,
        )


def test_paired_non_long_benchmark_accounts_are_rejected(
    development_replay,
) -> None:
    accounts = _accounts(DEVELOPMENT_STAGE)
    dates = pd.DatetimeIndex(
        ["2005-01-03", "2018-12-31"], name="date"
    )
    opens = pd.Series([100.0, 120.0], index=dates, dtype=float)
    targets = pd.Series([1, 0], index=dates, dtype=int)
    for cost in COST_ORDER:
        for policy in ("always_long", "aapl_buy_hold"):
            accounts[cost][policy] = run_continuous_ledger(
                opens,
                targets,
                policy_name=policy,
                cost_bps=COST_BPS[cost],
            ).state.to_checkpoint()

    with pytest.raises(
        ContextualExpertAggregationArtifactError,
        match="remain exactly LONG",
    ):
        build_composite_stage_checkpoint(
            stage=DEVELOPMENT_STAGE,
            replay_checkpoints={
                arm: development_replay.checkpoint for arm in ARM_ORDER
            },
            administrative_accounts=accounts,
        )


def test_seed_proof_tamper_and_stage_specific_null_misuse_are_rejected(
    development_replay,
    confirmation_forks,
) -> None:
    development = copy.deepcopy(_development_checkpoint(development_replay))
    development["arm_seed_equivalence"]["seed_target_source_policy"] = (
        GLOBAL_ONLY_ARM
    )
    development = _rehash_checkpoint(development)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(development)

    development = copy.deepcopy(_development_checkpoint(development_replay))
    development["arm_seed_equivalence"] = None
    development = _rehash_checkpoint(development)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(development)

    confirmation = copy.deepcopy(_confirmation_checkpoint(confirmation_forks))
    confirmation["arm_seed_equivalence"] = build_arm_seed_equivalence(
        _accounts(DEVELOPMENT_STAGE)
    )
    confirmation = _rehash_checkpoint(confirmation)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(confirmation)


def test_schema_and_checkpoint_reject_bool_integer_confusion(
    development_replay,
) -> None:
    with pytest.raises(ContextualExpertAggregationArtifactError):
        CanonicalTableSchema(
            columns=("x",), column_types=("integer",)
        )

    value = copy.deepcopy(_development_checkpoint(development_replay))
    value["checkpoint_schema_version"] = True
    value = _rehash_checkpoint(value)
    with pytest.raises(ContextualExpertAggregationArtifactError):
        parse_composite_stage_checkpoint(value)
