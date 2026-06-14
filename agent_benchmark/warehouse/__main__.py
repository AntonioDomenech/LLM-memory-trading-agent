from __future__ import annotations

import argparse
import json

from ..config_store import load_local_config
from .gdelt_events import download_gdelt_events, download_gdelt_events_parallel
from .macro import download_macro
from .news import download_news
from .prices import download_prices
from .sec import download_sec
from .store import Warehouse
from .universe import END_DATE, START_DATE, select_symbols
from .validate import validate_warehouse


def _symbols(args, *, include_context: bool = True):
    symbols = None
    if args.symbols:
        symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    return select_symbols(symbols, include_context=include_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical warehouse downloader")
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--start", default=START_DATE)
    bootstrap.add_argument("--end", default=END_DATE)

    prices = sub.add_parser("download-prices")
    prices.add_argument("--symbols", default="")
    prices.add_argument("--start", default=START_DATE)
    prices.add_argument("--end", default=END_DATE)
    prices.add_argument("--force", action="store_true")

    sec = sub.add_parser("download-sec")
    sec.add_argument("--symbols", default="")
    sec.add_argument("--user-agent", default="")

    macro = sub.add_parser("download-macro")
    macro.add_argument("--fred-key", default="")
    macro.add_argument("--start", default=START_DATE)
    macro.add_argument("--end", default=END_DATE)

    news = sub.add_parser("download-news")
    news.add_argument("--symbols", default="")
    news.add_argument("--start", default=START_DATE)
    news.add_argument("--end", default=END_DATE)
    news.add_argument("--max-months", type=int, default=None)
    news.add_argument("--max-records", type=int, default=100)
    news.add_argument("--sleep-seconds", type=float, default=5.2)
    news.add_argument("--min-interval-seconds", type=int, default=1)
    news.add_argument("--retry-until-success", action="store_true")
    news.add_argument("--max-retries", type=int, default=5)
    news.add_argument("--rate-limit-cooldown-seconds", type=float, default=300.0)
    news.add_argument("--max-backoff-seconds", type=float, default=3600.0)
    news.add_argument("--checkpoint-every", type=int, default=25)
    news.add_argument("--force", action="store_true")

    events = sub.add_parser("download-gdelt-events")
    events.add_argument("--symbols", default="")
    events.add_argument("--start", default=START_DATE)
    events.add_argument("--end", default="2016-12-31")
    events.add_argument("--sleep-seconds", type=float, default=1.0)
    events.add_argument("--retry-until-success", action="store_true")
    events.add_argument("--max-retries", type=int, default=5)
    events.add_argument("--rate-limit-cooldown-seconds", type=float, default=120.0)
    events.add_argument("--max-backoff-seconds", type=float, default=1800.0)
    events.add_argument("--checkpoint-every", type=int, default=25)
    events.add_argument("--chunksize", type=int, default=50000)
    events.add_argument("--workers", type=int, default=1)
    events.add_argument("--import-batch-size", type=int, default=16)
    events.add_argument("--keep-raw", action="store_true")
    events.add_argument("--keep-stage", action="store_true")
    events.add_argument("--force", action="store_true")

    full_news = sub.add_parser("download-news-full")
    full_news.add_argument("--symbols", default="")
    full_news.add_argument("--start", default=START_DATE)
    full_news.add_argument("--end", default=END_DATE)
    full_news.add_argument("--doc-start", default="2017-01-01")
    full_news.add_argument("--event-end", default="2016-12-31")
    full_news.add_argument("--event-sleep-seconds", type=float, default=1.0)
    full_news.add_argument("--event-workers", type=int, default=1)
    full_news.add_argument("--event-chunksize", type=int, default=100000)
    full_news.add_argument("--event-import-batch-size", type=int, default=16)
    full_news.add_argument("--doc-sleep-seconds", type=float, default=15.0)
    full_news.add_argument("--retry-until-success", action="store_true")
    full_news.add_argument("--rate-limit-cooldown-seconds", type=float, default=600.0)
    full_news.add_argument("--max-backoff-seconds", type=float, default=3600.0)
    full_news.add_argument("--checkpoint-every", type=int, default=25)
    full_news.add_argument("--keep-raw", action="store_true")
    full_news.add_argument("--keep-stage", action="store_true")
    full_news.add_argument("--force", action="store_true")

    sub.add_parser("validate")
    sub.add_parser("status")

    args = parser.parse_args()
    wh = Warehouse()
    try:
        if args.command == "bootstrap":
            result = wh.bootstrap(args.start, args.end)
        elif args.command == "download-prices":
            result = download_prices(wh, _symbols(args), start=args.start, end=args.end, force=args.force)
        elif args.command == "download-sec":
            local = load_local_config()
            user_agent = args.user_agent or local.secrets.sec_user_agent
            result = download_sec(wh, _symbols(args, include_context=False), user_agent=user_agent)
        elif args.command == "download-macro":
            local = load_local_config()
            fred_key = args.fred_key or local.secrets.fred_api_key
            result = download_macro(wh, api_key=fred_key, start=args.start, end=args.end)
        elif args.command == "download-news":
            result = download_news(
                wh,
                _symbols(args, include_context=False),
                start=args.start,
                end=args.end,
                max_months=args.max_months,
                max_records=args.max_records,
                sleep_seconds=args.sleep_seconds,
                min_interval_seconds=args.min_interval_seconds,
                retry_until_success=args.retry_until_success,
                max_retries=args.max_retries,
                rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
                max_backoff_seconds=args.max_backoff_seconds,
                checkpoint_every=args.checkpoint_every,
                force=args.force,
            )
        elif args.command == "download-gdelt-events":
            event_downloader = download_gdelt_events_parallel if args.workers > 1 else download_gdelt_events
            common = dict(
                warehouse=wh,
                symbols=_symbols(args, include_context=False),
                start=args.start,
                end=args.end,
                sleep_seconds=args.sleep_seconds,
                retry_until_success=args.retry_until_success,
                max_retries=args.max_retries,
                rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
                max_backoff_seconds=args.max_backoff_seconds,
                checkpoint_every=args.checkpoint_every,
                chunksize=args.chunksize,
                keep_raw=args.keep_raw,
                force=args.force,
            )
            if args.workers > 1:
                common["workers"] = args.workers
                common["keep_stage"] = args.keep_stage
                common["import_batch_size"] = args.import_batch_size
            result = event_downloader(
                **common,
            )
        elif args.command == "download-news-full":
            selected = _symbols(args, include_context=False)
            if args.event_workers > 1:
                event_result = download_gdelt_events_parallel(
                    wh,
                    selected,
                    start=args.start,
                    end=min(args.event_end, args.end),
                    workers=args.event_workers,
                    sleep_seconds=args.event_sleep_seconds,
                    retry_until_success=args.retry_until_success,
                    rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
                    max_backoff_seconds=args.max_backoff_seconds,
                    checkpoint_every=args.checkpoint_every,
                    chunksize=args.event_chunksize,
                    import_batch_size=args.event_import_batch_size,
                    keep_raw=args.keep_raw,
                    keep_stage=args.keep_stage,
                    force=args.force,
                )
            else:
                event_result = download_gdelt_events(
                    wh,
                    selected,
                    start=args.start,
                    end=min(args.event_end, args.end),
                    sleep_seconds=args.event_sleep_seconds,
                    retry_until_success=args.retry_until_success,
                    rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
                    max_backoff_seconds=args.max_backoff_seconds,
                    checkpoint_every=args.checkpoint_every,
                    chunksize=args.event_chunksize,
                    keep_raw=args.keep_raw,
                    force=args.force,
                )
            doc_result = download_news(
                wh,
                selected,
                start=max(args.doc_start, args.start),
                end=args.end,
                max_records=250,
                sleep_seconds=args.doc_sleep_seconds,
                min_interval_seconds=1,
                retry_until_success=args.retry_until_success,
                rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
                max_backoff_seconds=args.max_backoff_seconds,
                checkpoint_every=args.checkpoint_every,
                force=args.force,
            )
            result = {"gdelt_events": event_result, "gdelt_doc": doc_result}
        elif args.command == "validate":
            result = validate_warehouse(wh)
        elif args.command == "status":
            result = wh.status()
        else:
            raise ValueError(args.command)
        print(json.dumps(result, indent=2, default=str))
    finally:
        wh.close()


if __name__ == "__main__":
    main()
