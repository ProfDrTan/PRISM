"""
PRISM — run_daily.py
====================
Daily batch scheduler. Runs full analysis for all watchlist tickers
and generates LLM reports. Designed to trigger at 09:30 ET (US market open)
and 16:30 ET (US market close).

Usage:
  python run_daily.py --once                   # run immediately once
  python run_daily.py --tickers AAPL,MSFT,^GSPC
  python run_daily.py --schedule               # run on schedule (blocking)

Author: Professor Dr. Teik Kheong Tan
Built with: Claude (Anthropic)
"""

import os
import time
import argparse
import schedule

from prism_core import PRISMSystem, DEFAULT_TICKERS, normalise_ticker
from llm_insights import generate_llm_insights

DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


def run_batch(tickers: list[str], generate_llm: bool = True, quiet: bool = False):
    """
    Run PRISM analysis for all tickers in the list.
    Optionally generates LLM insights after market data analysis.
    """
    print(f"\n📡 PRISM Daily Batch — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   LLM reports: {'enabled' if generate_llm else 'disabled'}")
    print(f"   Demo mode: {DEMO_MODE}\n")

    system  = PRISMSystem()
    reports = []

    for ticker in tickers:
        symbol = normalise_ticker(ticker)
        try:
            report = system.run_one_ticker(symbol, quiet=quiet)
            if report:
                reports.append(report)

                if generate_llm and not DEMO_MODE:
                    print(f"  🤖 Generating LLM insights for {symbol}…")
                    try:
                        generate_llm_insights(report)
                        print(f"  ✓ LLM insights saved for {symbol}")
                    except Exception as e:
                        print(f"  ⚠ LLM insights failed for {symbol}: {e}")

        except Exception as e:
            print(f"  ✗ Failed {symbol}: {e}")

    if reports:
        path = system.save_session_summary(reports)
        print(f"\n✅ Session summary saved: {path}")
    else:
        print("\n⚠ No reports generated.")

    print(f"📡 Batch complete — {len(reports)}/{len(tickers)} tickers processed.\n")


def schedule_batch(tickers: list[str]):
    """
    Run on schedule: 09:30 ET and 16:30 ET daily.
    ET = UTC-4 (EDT) / UTC-5 (EST).
    Using UTC 13:30 (≈09:30 ET) and UTC 20:30 (≈16:30 ET).
    """
    print("📡 PRISM Scheduler starting…")
    print("   Scheduled runs: 13:30 UTC and 20:30 UTC daily")

    schedule.every().day.at("13:30").do(run_batch, tickers=tickers, generate_llm=True)
    schedule.every().day.at("20:30").do(run_batch, tickers=tickers, generate_llm=True)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Daily Batch Runner")
    parser.add_argument("--tickers",   default=",".join(DEFAULT_TICKERS),
                        help="Comma-separated ticker list")
    parser.add_argument("--once",      action="store_true",
                        help="Run immediately once and exit")
    parser.add_argument("--schedule",  action="store_true",
                        help="Run on daily schedule (blocking)")
    parser.add_argument("--no-llm",    action="store_true",
                        help="Skip LLM report generation")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if args.once:
        run_batch(tickers, generate_llm=not args.no_llm, quiet=args.quiet)
    elif args.schedule:
        schedule_batch(tickers)
    else:
        # Default: run once
        run_batch(tickers, generate_llm=not args.no_llm, quiet=args.quiet)
