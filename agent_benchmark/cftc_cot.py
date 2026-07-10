"""Point-in-time CFTC Legacy Commitments of Traders data helpers.

The module deliberately has no warehouse or benchmark-engine dependencies.  It
retrieves only three fixed financial-futures contracts from the CFTC's public
Socrata ``Legacy - Futures Only`` view and keeps the original response bytes so
that a run can prove exactly which payload it used.

Historical COT downloads are revised in place by the CFTC.  The public API does
not expose every original vintage.  Weeks covered by an official correction,
publication outage, or known stale-position announcement are therefore treated
as unavailable instead of allowing a backtest to consume today's corrected
history as though it had been known at the time.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlencode

import requests


CFTC_LEGACY_FUTURES_ONLY_DATASET_ID = "6dca-aqww"
CFTC_LEGACY_FUTURES_ONLY_CSV_URL = (
    f"https://publicreporting.cftc.gov/resource/{CFTC_LEGACY_FUTURES_ONLY_DATASET_ID}.csv"
)
CFTC_LEGACY_DATASET_PAGE = (
    "https://publicreporting.cftc.gov/Commitments-of-Traders/"
    f"Legacy-Futures-Only/{CFTC_LEGACY_FUTURES_ONLY_DATASET_ID}"
)
CFTC_SPECIAL_ANNOUNCEMENTS_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
    "HistoricalSpecialAnnouncements/index.htm"
)
CFTC_2019_SHUTDOWN_URL = "https://www.cftc.gov/PressRoom/PressReleases/7864-19"
CFTC_2023_ION_URL = "https://www.cftc.gov/PressRoom/PressReleases/8662-23"

# A report normally describes Tuesday's positions and is published on Friday.
# Eight calendar days is intentionally conservative: it does not expose the
# value to a simulated decision until the following Wednesday.
POINT_IN_TIME_RELEASE_LAG_DAYS = 8
MAX_SIGNAL_STALENESS_DAYS = 14
POST_2023_GUARD_DATE = date(2023, 12, 31)


@dataclass(frozen=True)
class COTMarket:
    key: str
    contract_code: str
    first_report_date: date
    description: str


COT_MARKETS: tuple[COTMarket, ...] = (
    COTMarket(
        key="sp500_emini",
        contract_code="13874A",
        first_report_date=date(1997, 9, 16),
        description="CME E-mini S&P 500 futures",
    ),
    COTMarket(
        key="nasdaq100_mini",
        contract_code="209742",
        first_report_date=date(1999, 6, 22),
        description="CME Nasdaq-100 mini futures",
    ),
    COTMarket(
        key="vix",
        contract_code="1170E1",
        first_report_date=date(2004, 7, 27),
        description="CBOE VIX futures",
    ),
)
COT_MARKETS_BY_CODE: Mapping[str, COTMarket] = {
    market.contract_code: market for market in COT_MARKETS
}
COT_MARKETS_BY_KEY: Mapping[str, COTMarket] = {market.key: market for market in COT_MARKETS}
SUPPORTED_CONTRACT_CODES: tuple[str, ...] = tuple(
    market.contract_code for market in COT_MARKETS
)
EARLIEST_SUPPORTED_REPORT_DATE = min(market.first_report_date for market in COT_MARKETS)


class COTDataError(ValueError):
    """Raised when a COT request or payload violates the frozen data contract."""


class Post2023AccessError(COTDataError):
    """Raised when post-2023 data is touched without an explicit opt-in."""


@dataclass(frozen=True)
class OfficialCOTAnomaly:
    anomaly_id: str
    kind: str
    contract_codes: frozenset[str]
    reason: str
    source_url: str
    report_dates: frozenset[date] = frozenset()
    start_date: date | None = None
    end_date: date | None = None
    action: str = "drop"

    def matches(self, contract_code: str, report_date: date) -> bool:
        if contract_code not in self.contract_codes:
            return False
        if self.report_dates and report_date in self.report_dates:
            return True
        return bool(
            self.start_date is not None
            and self.end_date is not None
            and self.start_date <= report_date <= self.end_date
        )


_ALL_CODES = frozenset(SUPPORTED_CONTRACT_CODES)

# This is intentionally explicit rather than inferred from gaps in the revised
# dataset.  Each entry is backed by a CFTC announcement.  The current public
# archive does not retain the original vintages needed to reconstruct what a
# trader saw, so the safe point-in-time action is always ``drop``.
OFFICIAL_ANOMALY_CALENDAR: tuple[OfficialCOTAnomaly, ...] = (
    OfficialCOTAnomaly(
        anomaly_id="cftc-vix-corrections-2009",
        kind="corrected_after_publication",
        contract_codes=frozenset({"1170E1"}),
        report_dates=frozenset(
            {date(2009, 11, 17), date(2009, 11, 24), date(2009, 12, 1)}
        ),
        reason="CFTC corrected VIX large-trader reporting errors after publication.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-nasdaq-clearing-adjustment-2011-06-21",
        kind="exchange_open_interest_adjustment",
        contract_codes=frozenset({"209742"}),
        report_dates=frozenset({date(2011, 6, 21)}),
        reason="CFTC adjusted Nasdaq-100 mini open interest after a clearing-system problem.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-vix-sandy-stale-positions-2012-10-30",
        kind="stale_positions",
        contract_codes=frozenset({"1170E1"}),
        report_dates=frozenset({date(2012, 10, 30)}),
        reason="The dated VIX report actually reflected positions from 2012-10-26.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-account-reclassification-2012-11-27",
        kind="corrected_after_publication",
        contract_codes=_ALL_CODES,
        report_dates=frozenset({date(2012, 11, 27)}),
        reason="CFTC republished reports after account transfers were initially misclassified.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-appropriations-lapse-2018-2019",
        kind="delayed_publication",
        contract_codes=_ALL_CODES,
        start_date=date(2018, 12, 24),
        # At two catch-up releases per week, the 2019-02-19 report was still
        # published after report_date + 8 days.  The 2019-02-26 report was
        # available before that conservative boundary.
        end_date=date(2019, 2, 19),
        reason="Weekly COT publication was suspended and later released out of schedule.",
        source_url=CFTC_2019_SHUTDOWN_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-reporting-firm-correction-2019-03-26",
        kind="corrected_after_publication",
        contract_codes=_ALL_CODES,
        report_dates=frozenset({date(2019, 3, 26)}),
        reason="CFTC added reportable positions to all three selected contracts after publication.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-ion-publication-outage-2023",
        kind="delayed_publication",
        contract_codes=_ALL_CODES,
        start_date=date(2023, 1, 31),
        end_date=date(2023, 3, 14),
        reason="The ION cyber incident delayed and disrupted sequential COT publication.",
        source_url=CFTC_2023_ION_URL,
    ),
    OfficialCOTAnomaly(
        anomaly_id="cftc-appropriations-lapse-2025",
        kind="delayed_publication",
        contract_codes=_ALL_CODES,
        start_date=date(2025, 9, 30),
        end_date=date(2025, 12, 23),
        reason="COT processing and publication were interrupted during a funding lapse.",
        source_url=CFTC_SPECIAL_ANNOUNCEMENTS_URL,
    ),
)


def anomaly_calendar_sha256() -> str:
    """Return a stable hash suitable for an immutable run manifest."""

    payload = []
    for anomaly in OFFICIAL_ANOMALY_CALENDAR:
        item = asdict(anomaly)
        item["contract_codes"] = sorted(anomaly.contract_codes)
        item["report_dates"] = sorted(value.isoformat() for value in anomaly.report_dates)
        item["start_date"] = anomaly.start_date.isoformat() if anomaly.start_date else None
        item["end_date"] = anomaly.end_date.isoformat() if anomaly.end_date else None
        payload.append(item)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def official_anomaly_for(
    contract_code: str, report_date: date | str
) -> OfficialCOTAnomaly | None:
    code = _validate_contract_code(contract_code)
    parsed_date = _coerce_date(report_date, field_name="report_date")
    for anomaly in OFFICIAL_ANOMALY_CALENDAR:
        if anomaly.matches(code, parsed_date):
            return anomaly
    return None


@dataclass(frozen=True)
class COTRecord:
    source_id: str
    market_name: str
    contract_code: str
    report_date: date
    open_interest: int
    noncommercial_long: int
    noncommercial_short: int
    anomaly_id: str | None = None
    anomaly_reason: str | None = None
    anomaly_source_url: str | None = None

    @property
    def available_date(self) -> date:
        return self.report_date + timedelta(days=POINT_IN_TIME_RELEASE_LAG_DAYS)

    @property
    def noncommercial_net(self) -> int:
        return self.noncommercial_long - self.noncommercial_short

    @property
    def noncommercial_net_share(self) -> float:
        return self.noncommercial_net / self.open_interest

    @property
    def usable(self) -> bool:
        return self.anomaly_id is None


@dataclass(frozen=True)
class COTExclusion:
    source_id: str
    contract_code: str
    report_date: date
    anomaly_id: str
    reason: str
    source_url: str


@dataclass(frozen=True)
class CleanCOTRecords:
    records: tuple[COTRecord, ...]
    exclusions: tuple[COTExclusion, ...]


@dataclass(frozen=True)
class COTFeature:
    contract_code: str
    market_key: str
    report_date: date
    available_date: date
    age_since_available_days: int
    noncommercial_net: int
    noncommercial_net_share: float


@dataclass(frozen=True)
class COTRetrievalMetadata:
    dataset_id: str
    dataset_page: str
    source_url: str
    requested_start_date: str
    requested_end_date: str
    contract_codes: tuple[str, ...]
    retrieved_at_utc: str
    raw_response_sha256: str
    raw_size_bytes: int
    http_status: int
    content_type: str
    count_source_url: str
    count_response_sha256: str
    count_size_bytes: int
    count_http_status: int
    count_content_type: str
    anomaly_calendar_sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class COTDownload:
    raw_csv: bytes
    count_proof_csv: bytes
    metadata: COTRetrievalMetadata


_CSV_FIELDS = (
    "id",
    "market_and_exchange_names",
    "report_date_as_yyyy_mm_dd",
    "cftc_contract_market_code",
    "open_interest_all",
    "noncomm_positions_long_all",
    "noncomm_positions_short_all",
)

_COUNT_PROOF_FIELD = "row_count"


def _coerce_date(value: date | datetime | str, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise COTDataError(f"{field_name} is required")
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise COTDataError(f"{field_name} must be an ISO date: {value!r}") from exc


def _validate_contract_code(value: str) -> str:
    code = str(value).strip().upper()
    if code not in COT_MARKETS_BY_CODE:
        raise COTDataError(
            f"Unsupported CFTC contract code {value!r}; allowed codes are "
            f"{', '.join(SUPPORTED_CONTRACT_CODES)}"
        )
    return code


def _validate_access_date(value: date, *, allow_post_2023: bool, field_name: str) -> None:
    if value > POST_2023_GUARD_DATE and not allow_post_2023:
        raise Post2023AccessError(
            f"{field_name} {value.isoformat()} is after 2023-12-31; pass "
            "allow_post_2023=True to make that access explicit"
        )


def validate_request_bounds(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
) -> tuple[date, date]:
    start = _coerce_date(start_date, field_name="start_date")
    end = _coerce_date(end_date, field_name="end_date")
    if start > end:
        raise COTDataError("start_date must be on or before end_date")
    if start < EARLIEST_SUPPORTED_REPORT_DATE:
        raise COTDataError(
            "start_date precedes the first selected contract's supported history "
            f"({EARLIEST_SUPPORTED_REPORT_DATE.isoformat()})"
        )
    _validate_access_date(start, allow_post_2023=allow_post_2023, field_name="start_date")
    _validate_access_date(end, allow_post_2023=allow_post_2023, field_name="end_date")
    return start, end


def build_socrata_params(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
) -> dict[str, str]:
    """Build the bounded fixed-ID query sent to the official CFTC view."""

    start, end = validate_request_bounds(
        start_date, end_date, allow_post_2023=allow_post_2023
    )
    end_exclusive = end + timedelta(days=1)
    quoted_codes = ",".join(f"'{code}'" for code in SUPPORTED_CONTRACT_CODES)
    where = (
        f"cftc_contract_market_code in ({quoted_codes}) "
        f"AND report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000' "
        f"AND report_date_as_yyyy_mm_dd < '{end_exclusive.isoformat()}T00:00:00.000'"
    )
    return {
        "$select": ",".join(_CSV_FIELDS),
        "$where": where,
        "$order": "report_date_as_yyyy_mm_dd ASC,cftc_contract_market_code ASC,id ASC",
        "$limit": "50000",
    }


def build_socrata_count_params(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
) -> dict[str, str]:
    """Build an independent bounded row-count query for completeness proof."""

    data_params = build_socrata_params(
        start_date,
        end_date,
        allow_post_2023=allow_post_2023,
    )
    return {
        "$select": f"count(*) AS {_COUNT_PROOF_FIELD}",
        "$where": data_params["$where"],
        "$limit": "1",
    }


def build_socrata_csv_url(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
) -> str:
    params = build_socrata_params(
        start_date, end_date, allow_post_2023=allow_post_2023
    )
    return f"{CFTC_LEGACY_FUTURES_ONLY_CSV_URL}?{urlencode(params)}"


def build_socrata_count_url(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
) -> str:
    params = build_socrata_count_params(
        start_date,
        end_date,
        allow_post_2023=allow_post_2023,
    )
    return f"{CFTC_LEGACY_FUTURES_ONLY_CSV_URL}?{urlencode(params)}"


def _retrieval_timestamp(value: datetime | None) -> str:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise COTDataError("retrieved_at must be timezone-aware")
    return (
        observed.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def fetch_legacy_cot_csv(
    start_date: date | str,
    end_date: date | str,
    *,
    allow_post_2023: bool = False,
    timeout_seconds: float = 30.0,
    session: object = requests,
    retrieved_at: datetime | None = None,
) -> COTDownload:
    """Retrieve the official CSV plus an independent exact row-count proof."""

    start, end = validate_request_bounds(
        start_date, end_date, allow_post_2023=allow_post_2023
    )
    retrieval_timestamp = _retrieval_timestamp(retrieved_at)
    params = build_socrata_params(start, end, allow_post_2023=allow_post_2023)
    response = session.get(  # type: ignore[attr-defined]
        CFTC_LEGACY_FUTURES_ONLY_CSV_URL,
        params=params,
        headers={"Accept": "text/csv", "User-Agent": "LLM-memory-trading-agent/1.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    raw = bytes(response.content)
    response_headers = getattr(response, "headers", {}) or {}
    status = int(getattr(response, "status_code", 200))
    source_url = str(
        getattr(response, "url", "")
        or build_socrata_csv_url(start, end, allow_post_2023=allow_post_2023)
    )
    count_params = build_socrata_count_params(
        start,
        end,
        allow_post_2023=allow_post_2023,
    )
    count_response = session.get(  # type: ignore[attr-defined]
        CFTC_LEGACY_FUTURES_ONLY_CSV_URL,
        params=count_params,
        headers={"Accept": "text/csv", "User-Agent": "LLM-memory-trading-agent/1.0"},
        timeout=timeout_seconds,
    )
    count_response.raise_for_status()
    count_raw = bytes(count_response.content)
    count_response_headers = getattr(count_response, "headers", {}) or {}
    count_status = int(getattr(count_response, "status_code", 200))
    count_source_url = str(
        getattr(count_response, "url", "")
        or build_socrata_count_url(start, end, allow_post_2023=allow_post_2023)
    )
    metadata = COTRetrievalMetadata(
        dataset_id=CFTC_LEGACY_FUTURES_ONLY_DATASET_ID,
        dataset_page=CFTC_LEGACY_DATASET_PAGE,
        source_url=source_url,
        requested_start_date=start.isoformat(),
        requested_end_date=end.isoformat(),
        contract_codes=SUPPORTED_CONTRACT_CODES,
        retrieved_at_utc=retrieval_timestamp,
        raw_response_sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        raw_size_bytes=len(raw),
        http_status=status,
        content_type=str(response_headers.get("content-type", "")),
        count_source_url=count_source_url,
        count_response_sha256=f"sha256:{hashlib.sha256(count_raw).hexdigest()}",
        count_size_bytes=len(count_raw),
        count_http_status=count_status,
        count_content_type=str(count_response_headers.get("content-type", "")),
        anomaly_calendar_sha256=anomaly_calendar_sha256(),
    )
    return COTDownload(raw_csv=raw, count_proof_csv=count_raw, metadata=metadata)


def _parse_nonnegative_integer(value: object, *, field_name: str, row_number: int) -> int:
    text = str(value).strip()
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise COTDataError(
            f"row {row_number}: {field_name} must be an integer, got {value!r}"
        ) from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value() or parsed < 0:
        raise COTDataError(
            f"row {row_number}: {field_name} must be a non-negative integer, got {value!r}"
        )
    return int(parsed)


def parse_socrata_count_proof_csv(raw_csv: bytes | str) -> int:
    """Parse the one-row aggregate returned by the independent count query."""

    if isinstance(raw_csv, bytes):
        try:
            text = raw_csv.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise COTDataError("CFTC count proof is not valid UTF-8") from exc
    else:
        text = str(raw_csv).lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != (_COUNT_PROOF_FIELD,):
        raise COTDataError(
            f"CFTC count proof must contain only the {_COUNT_PROOF_FIELD!r} field"
        )
    rows = list(reader)
    if len(rows) != 1:
        raise COTDataError("CFTC count proof must contain exactly one aggregate row")
    return _parse_nonnegative_integer(
        rows[0].get(_COUNT_PROOF_FIELD),
        field_name=_COUNT_PROOF_FIELD,
        row_number=2,
    )


def parse_legacy_cot_csv(
    raw_csv: bytes | str,
    *,
    allow_post_2023: bool = False,
    expected_start_date: date | str | None = None,
    expected_end_date: date | str | None = None,
) -> tuple[COTRecord, ...]:
    """Parse and validate a CFTC CSV payload without performing network I/O."""

    if (expected_start_date is None) != (expected_end_date is None):
        raise COTDataError(
            "expected_start_date and expected_end_date must be provided together"
        )
    expected_bounds = None
    if expected_start_date is not None and expected_end_date is not None:
        expected_bounds = validate_request_bounds(
            expected_start_date,
            expected_end_date,
            allow_post_2023=allow_post_2023,
        )

    if isinstance(raw_csv, bytes):
        try:
            text = raw_csv.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise COTDataError("CFTC CSV is not valid UTF-8") from exc
    else:
        text = str(raw_csv).lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    present_fields = set(reader.fieldnames or ())
    missing = sorted(set(_CSV_FIELDS) - present_fields)
    if missing:
        raise COTDataError(f"CFTC CSV is missing required fields: {', '.join(missing)}")

    records: list[COTRecord] = []
    seen_keys: set[tuple[str, date]] = set()
    for row_number, row in enumerate(reader, start=2):
        source_id = str(row.get("id") or "").strip()
        if not source_id:
            raise COTDataError(f"row {row_number}: id is required")
        code = _validate_contract_code(str(row.get("cftc_contract_market_code") or ""))
        report_date = _coerce_date(
            str(row.get("report_date_as_yyyy_mm_dd") or ""), field_name="report_date"
        )
        _validate_access_date(
            report_date,
            allow_post_2023=allow_post_2023,
            field_name=f"row {row_number} report_date",
        )
        if expected_bounds is not None and not (
            expected_bounds[0] <= report_date <= expected_bounds[1]
        ):
            raise COTDataError(
                f"row {row_number}: report_date {report_date.isoformat()} is outside "
                "the expected response bounds"
            )
        market = COT_MARKETS_BY_CODE[code]
        if report_date < market.first_report_date:
            raise COTDataError(
                f"row {row_number}: {code} predates its supported history "
                f"({market.first_report_date.isoformat()})"
            )
        key = (code, report_date)
        if key in seen_keys:
            raise COTDataError(
                f"row {row_number}: duplicate COT observation for {code} on {report_date}"
            )
        seen_keys.add(key)
        open_interest = _parse_nonnegative_integer(
            row.get("open_interest_all"), field_name="open_interest_all", row_number=row_number
        )
        if open_interest == 0:
            raise COTDataError(f"row {row_number}: open_interest_all must be greater than zero")
        noncommercial_long = _parse_nonnegative_integer(
            row.get("noncomm_positions_long_all"),
            field_name="noncomm_positions_long_all",
            row_number=row_number,
        )
        noncommercial_short = _parse_nonnegative_integer(
            row.get("noncomm_positions_short_all"),
            field_name="noncomm_positions_short_all",
            row_number=row_number,
        )
        anomaly = official_anomaly_for(code, report_date)
        records.append(
            COTRecord(
                source_id=source_id,
                market_name=str(row.get("market_and_exchange_names") or "").strip(),
                contract_code=code,
                report_date=report_date,
                open_interest=open_interest,
                noncommercial_long=noncommercial_long,
                noncommercial_short=noncommercial_short,
                anomaly_id=anomaly.anomaly_id if anomaly else None,
                anomaly_reason=anomaly.reason if anomaly else None,
                anomaly_source_url=anomaly.source_url if anomaly else None,
            )
        )
    records.sort(key=lambda item: (item.report_date, item.contract_code, item.source_id))
    return tuple(records)


def apply_official_anomaly_calendar(records: Iterable[COTRecord]) -> CleanCOTRecords:
    """Partition records, retaining an auditable reason for every dropped week."""

    usable: list[COTRecord] = []
    exclusions: list[COTExclusion] = []
    for record in records:
        anomaly = official_anomaly_for(record.contract_code, record.report_date)
        if anomaly is None:
            usable.append(record)
            continue
        exclusions.append(
            COTExclusion(
                source_id=record.source_id,
                contract_code=record.contract_code,
                report_date=record.report_date,
                anomaly_id=anomaly.anomaly_id,
                reason=anomaly.reason,
                source_url=anomaly.source_url,
            )
        )
    usable.sort(key=lambda item: (item.report_date, item.contract_code, item.source_id))
    exclusions.sort(key=lambda item: (item.report_date, item.contract_code, item.source_id))
    return CleanCOTRecords(records=tuple(usable), exclusions=tuple(exclusions))


def _validate_as_of(as_of: date | str, *, allow_post_2023: bool) -> date:
    parsed = _coerce_date(as_of, field_name="as_of")
    _validate_access_date(parsed, allow_post_2023=allow_post_2023, field_name="as_of")
    return parsed


def feature_history_as_of(
    records: Iterable[COTRecord],
    contract_code: str,
    as_of: date | str,
    *,
    allow_post_2023: bool = False,
) -> tuple[COTFeature, ...]:
    """Return only non-anomalous observations released by ``as_of``."""

    code = _validate_contract_code(contract_code)
    decision_date = _validate_as_of(as_of, allow_post_2023=allow_post_2023)
    materialized = tuple(records)
    for record in materialized:
        _validate_contract_code(record.contract_code)
        _validate_access_date(
            record.report_date,
            allow_post_2023=allow_post_2023,
            field_name=f"record {record.source_id} report_date",
        )
        if record.open_interest <= 0:
            raise COTDataError(
                f"record {record.source_id} open_interest must be greater than zero"
            )
    cleaned = apply_official_anomaly_calendar(materialized).records
    history = []
    for record in cleaned:
        if record.contract_code != code or record.available_date > decision_date:
            continue
        history.append(
            COTFeature(
                contract_code=code,
                market_key=COT_MARKETS_BY_CODE[code].key,
                report_date=record.report_date,
                available_date=record.available_date,
                age_since_available_days=(decision_date - record.available_date).days,
                noncommercial_net=record.noncommercial_net,
                noncommercial_net_share=record.noncommercial_net_share,
            )
        )
    history.sort(key=lambda item: item.report_date)
    return tuple(history)


def latest_features_as_of(
    records: Iterable[COTRecord],
    as_of: date | str,
    *,
    max_staleness_days: int = MAX_SIGNAL_STALENESS_DAYS,
    allow_post_2023: bool = False,
) -> dict[str, COTFeature | None]:
    """Return one point-in-time feature per fixed market, or ``None`` if stale.

    Staleness is counted from the conservative availability date.  A feature is
    usable through age 14 and neutralized starting at age 15.
    """

    if max_staleness_days < 0:
        raise COTDataError("max_staleness_days must be non-negative")
    decision_date = _validate_as_of(as_of, allow_post_2023=allow_post_2023)
    materialized = tuple(records)
    output: dict[str, COTFeature | None] = {}
    for code in SUPPORTED_CONTRACT_CODES:
        history = feature_history_as_of(
            materialized,
            code,
            decision_date,
            allow_post_2023=allow_post_2023,
        )
        latest = history[-1] if history else None
        if latest is None or latest.age_since_available_days > max_staleness_days:
            output[code] = None
        else:
            output[code] = latest
    return output


def trailing_net_share_zscore(
    history: Sequence[COTFeature],
    *,
    lookback_weeks: int,
) -> float | None:
    """Score the latest value against prior released weeks only.

    The current observation is never included in its own mean or dispersion.
    This helper performs no filling: an incomplete baseline or a constant
    baseline returns ``None`` rather than manufacturing a signal.
    """

    if lookback_weeks < 2:
        raise COTDataError("lookback_weeks must be at least 2")
    if len(history) < lookback_weeks + 1:
        return None
    ordered = sorted(history, key=lambda item: item.report_date)
    baseline = [
        item.noncommercial_net_share for item in ordered[-lookback_weeks - 1 : -1]
    ]
    current = ordered[-1].noncommercial_net_share
    if not all(math.isfinite(value) for value in [*baseline, current]):
        raise COTDataError("net-share history contains a non-finite value")
    dispersion = pstdev(baseline)
    if dispersion == 0.0:
        return None
    return (current - fmean(baseline)) / dispersion


__all__ = [
    "CFTC_LEGACY_FUTURES_ONLY_DATASET_ID",
    "CFTC_LEGACY_FUTURES_ONLY_CSV_URL",
    "CFTC_LEGACY_DATASET_PAGE",
    "CFTC_SPECIAL_ANNOUNCEMENTS_URL",
    "POINT_IN_TIME_RELEASE_LAG_DAYS",
    "MAX_SIGNAL_STALENESS_DAYS",
    "POST_2023_GUARD_DATE",
    "COT_MARKETS",
    "COT_MARKETS_BY_CODE",
    "COT_MARKETS_BY_KEY",
    "SUPPORTED_CONTRACT_CODES",
    "EARLIEST_SUPPORTED_REPORT_DATE",
    "OFFICIAL_ANOMALY_CALENDAR",
    "COTDataError",
    "Post2023AccessError",
    "COTMarket",
    "OfficialCOTAnomaly",
    "COTRecord",
    "COTExclusion",
    "CleanCOTRecords",
    "COTFeature",
    "COTRetrievalMetadata",
    "COTDownload",
    "anomaly_calendar_sha256",
    "official_anomaly_for",
    "validate_request_bounds",
    "build_socrata_params",
    "build_socrata_csv_url",
    "fetch_legacy_cot_csv",
    "parse_legacy_cot_csv",
    "apply_official_anomaly_calendar",
    "feature_history_as_of",
    "latest_features_as_of",
    "trailing_net_share_zscore",
]
