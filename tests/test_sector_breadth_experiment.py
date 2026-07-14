from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from agent_benchmark import sector_breadth_experiment as experiment


def _price_rows(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "aapl_open": [100.0 + index for index in range(len(dates))],
            "aapl_close": [101.0 + index for index in range(len(dates))],
            "aapl_adj_close": [50.0 + index for index in range(len(dates))],
            "spy_adj_close": [200.0 + index for index in range(len(dates))],
            "qqq_adj_close": [150.0 + index for index in range(len(dates))],
        }
    )


def _context(index: list[str]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index)
    return pd.DataFrame(
        {
            name: [100.0 + position for position in range(len(dates))]
            for name in (
                "xlb_adj_close",
                "xle_adj_close",
                "xlf_adj_close",
                "xli_adj_close",
                "xlk_adj_close",
                "xlp_adj_close",
                "xlu_adj_close",
                "xlv_adj_close",
                "xly_adj_close",
                "iwm_adj_close",
                "vix_close",
            )
        },
        index=dates,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(path: Path) -> None:
    _git(path, "init")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "Tests")


def _unselected_manifest() -> dict[str, object]:
    return experiment._manifest_with_hash(
        {
            "contract_version": experiment.CONTRACT_VERSION,
            "stage": "development",
            "development_pass": False,
            "intermediate_validation_pass": False,
            "final_selection_frozen": False,
            "selected_candidate_id": None,
            "explicit_no_winner_gate": {
                "triggered": True,
                "reason": "no_candidate_passed",
            },
            "frozen_policy": {
                "final_filename": None,
                "final_sha256": None,
            },
        }
    )


def _selected_manifest(policy_bytes: bytes) -> dict[str, object]:
    return experiment._manifest_with_hash(
        {
            "contract_version": experiment.CONTRACT_VERSION,
            "stage": "development",
            "development_pass": True,
            "intermediate_validation_pass": True,
            "final_selection_frozen": True,
            "selected_candidate_id": "sector_breadth_p55_e0",
            "explicit_no_winner_gate": {"triggered": False, "reason": None},
            "frozen_policy": {
                "final_filename": "final_frozen_policy_through_2023.json",
                "final_sha256": experiment._sha256_tagged(policy_bytes),
            },
        }
    )


def test_development_price_loader_returns_no_post_cutoff_row(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    _price_rows(
        [
            "1999-01-04",
            "2023-12-29",
            "2024-01-02",
            "2026-07-09",
        ]
    ).to_csv(source, index=False)

    frame, provenance = experiment.load_development_price_csv(source)

    assert list(frame.index.strftime("%Y-%m-%d")) == ["1999-01-04", "2023-12-29"]
    assert frame.index.max() <= experiment.DEVELOPMENT_DATA_END
    assert provenance["bounded_last_date"] == "2023-12-29"
    assert provenance["post_2023_rows_returned"] is False
    assert "2023-12-31" in provenance["query_parameters"]


@pytest.mark.parametrize("kind", ["price", "context"])
def test_injected_development_inputs_reject_post_cutoff_rows(kind: str) -> None:
    prices = _price_rows(["1999-01-04", "2023-12-29"])
    context = _context(["2000-05-26", "2023-12-29"])
    if kind == "price":
        prices = pd.concat(
            [prices, _price_rows(["2024-01-02"])], ignore_index=True
        ).sort_values("date")
    else:
        context = pd.concat([context, _context(["2024-01-02"])]).sort_index()

    with pytest.raises(experiment.SectorBreadthExperimentError, match="post-2023"):
        experiment._bounded_development_inputs(prices, context)


def test_development_rejects_a_wholly_missing_context_session() -> None:
    prices = _price_rows(
        ["1999-01-04", "2000-05-26", "2000-05-30", "2023-12-29"]
    )
    context = _context(["2000-05-26", "2023-12-29"])

    with pytest.raises(
        experiment.SectorBreadthExperimentError, match="exactly equal every expected price session"
    ):
        experiment._bounded_development_inputs(prices, context)


def test_validation_gate_requires_positive_edge_in_every_negative_buy_hold_year() -> None:
    failed = experiment._negative_buy_hold_year_gate(
        {"2019": 0.01, "2020": -0.001, "2021": -0.02},
        {"2019": 0.20, "2020": -0.10, "2021": 0.15},
    )
    passed = experiment._negative_buy_hold_year_gate(
        {"2019": -0.02, "2020": 0.001, "2021": -0.03},
        {"2019": 0.20, "2020": -0.10, "2021": 0.15},
    )

    assert failed["negative_buy_hold_years"] == ["2020"]
    assert failed["all_negative_buy_hold_years_have_positive_active_log_edge"] is False
    assert passed["all_negative_buy_hold_years_have_positive_active_log_edge"] is True


def test_context_public_loader_receives_exact_pre2024_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_path = tmp_path / "prices.csv"
    context_path = tmp_path / "context.parquet"
    price_path.write_bytes(b"price-source")
    context_path.write_bytes(b"context-source")
    prices = experiment.canonical_context_frame(
        _price_rows(["1999-01-04", "2023-12-29"])
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        experiment,
        "load_development_price_csv",
        lambda path: (prices, {"source_path": str(path)}),
    )

    def fake_context(path: Path, *, start: object, end: object) -> pd.DataFrame:
        observed.update(path=path, start=pd.Timestamp(start), end=pd.Timestamp(end))
        return _context(["2000-05-26", "2023-12-29"])

    monkeypatch.setattr(experiment, "load_sector_context_parquet", fake_context)

    loaded = experiment.load_public_development_inputs(
        price_artifact=price_path, context_parquet=context_path
    )

    assert observed == {
        "path": context_path,
        "start": experiment.CONTEXT_INPUT_START,
        "end": experiment.DEVELOPMENT_DATA_END,
    }
    assert loaded.context_frame.index.max() <= experiment.DEVELOPMENT_DATA_END
    assert loaded.provenance["context"]["post_2023_rows_returned"] is False
    assert loaded.provenance["context"]["observed_filesystem_creation_time_utc"]
    assert loaded.provenance["context"]["observed_filesystem_last_write_time_utc"]
    assert loaded.provenance["context"]["download_log"]["present"] is False


def test_final_rejects_uncommitted_manifest_before_market_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    tracked = tmp_path / "README.md"
    tracked.write_text("test\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "initial")
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    market_loaded = False

    def forbidden_market_load(**kwargs: object) -> tuple[object, ...]:
        nonlocal market_loaded
        market_loaded = True
        raise AssertionError("holdout loader must not be reached")

    monkeypatch.setattr(experiment, "_load_final_inputs", forbidden_market_load)

    with pytest.raises(experiment.SectorBreadthExperimentError, match="clean branch"):
        experiment.run_final_experiment(
            repo_root=tmp_path,
            selection_manifest=manifest,
            price_artifact=tmp_path / "holdout-prices.csv",
            context_parquet=tmp_path / "holdout-context.parquet",
            output_dir=tmp_path / "final-output",
        )
    assert market_loaded is False


def test_final_rejects_exact_committed_no_winner_manifest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_bytes(
        (json.dumps(_unselected_manifest(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    _git(tmp_path, "add", "selection_manifest.json")
    _git(tmp_path, "commit", "-m", "freeze rejected selection")

    with pytest.raises(
        experiment.SectorBreadthExperimentError, match="unselected or failed"
    ):
        experiment.validate_final_selection_manifest(
            repo_root=tmp_path, selection_manifest=manifest
        )


def test_final_accepts_only_exact_committed_selected_manifest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    bundle = tmp_path / "e" / "selected"
    bundle.mkdir(parents=True)
    policy_bytes = b'{"frozen":"policy"}\n'
    (bundle / ".gitattributes").write_bytes(b"* -text\n")
    (bundle / "final_frozen_policy_through_2023.json").write_bytes(policy_bytes)
    manifest = bundle / "selection_manifest.json"
    manifest.write_bytes(
        (
            json.dumps(_selected_manifest(policy_bytes), indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )
    _git(tmp_path, "add", "e/selected")
    _git(tmp_path, "commit", "-m", "freeze selected policy")

    validated = experiment.validate_final_selection_manifest(
        repo_root=tmp_path, selection_manifest=manifest
    )

    assert validated.manifest["selected_candidate_id"] == "sector_breadth_p55_e0"
    assert validated.git_branch
