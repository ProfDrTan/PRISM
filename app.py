"""
PRISM — app.py
==============
Flask REST API server. Entry point for the Hugging Face Space.

Serves:
  GET  /                          → Dashboard HTML
  GET  /api/analyze?ticker=X      → Full PRISM analysis (JSON)
  GET  /api/history/<ticker>      → OHLCV time-series for TradingView
  GET  /api/llm-insights?ticker=X → Cached LLM insights panel
  POST /api/feedback              → Log user feedback
  GET  /api/feedback/status       → Feedback diagnostics
  GET  /api/admin/feedback        → Download feedback log
  GET  /api/session-summary/latest → Latest session summary
  GET  /api/tickers               → Available ticker list

Author: Professor Dr. Teik Kheong Tan
Built with: Claude (Anthropic)
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from datetime import datetime, timezone
from pathlib import Path

from storage_paths import resolve_data_dir
from prism_core import (
    PRISMSystem, normalise_ticker, DEFAULT_TICKERS, TICKER_LABELS,
)
from llm_insights import load_cached_llm_insights, generate_llm_insights

# ── Environment ────────────────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEMO_MODE    = os.environ.get("DEMO_MODE", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = resolve_data_dir(PROJECT_ROOT, DEMO_MODE)
SESSIONS_DIR = DATA_DIR / "sessions"
YF_CACHE_DIR = DATA_DIR / "yfinance_tz_cache"

# ── Timeframe map ──────────────────────────────────────────────────────────────
TIMEFRAME_MAP = {
    "1D":  ("1d",  "2m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "6M":  ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y":  ("1y",  "1d"),
    "5Y":  ("5y",  "1wk"),
}

# ── Initialise PRISM engine ────────────────────────────────────────────────────
try:
    prism = PRISMSystem()
except Exception as e:
    print(f"⚠ PRISM engine failed to initialise: {e}")
    prism = None

app = Flask(__name__)
CORS(app)

DATA_DIR.mkdir(parents=True, exist_ok=True)
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _feedback_log_path() -> Path:
    if DEMO_MODE:
        p = PROJECT_ROOT / "data" / "demo_guests"
        p.mkdir(parents=True, exist_ok=True)
        return p / "feedback_logs.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "feedback_logs.json"


def _index_to_unix(index_values) -> np.ndarray:
    if isinstance(index_values, pd.DatetimeIndex):
        idx = index_values
    else:
        idx = pd.to_datetime(index_values, utc=True, errors="coerce")
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return np.asarray(idx.asi8, dtype=np.int64) // 10**9


def _normalise_download_frame(df, symbol_token: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol_token in df.columns.get_level_values(-1):
                return df.xs(symbol_token, axis=1, level=-1, drop_level=True)
        except Exception:
            pass
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the PRISM dashboard."""
    return render_template("index.html")


@app.route("/api/tickers")
def get_tickers():
    """Return the list of available tickers with human-readable labels."""
    return jsonify([
        {"symbol": t, "label": TICKER_LABELS.get(t, t)}
        for t in DEFAULT_TICKERS
    ])


@app.route("/api/analyze")
def analyze_ticker():
    """
    Run a full PRISM analysis for a ticker.
    Query params: ticker, timeframe (optional), period, interval
    """
    if not prism:
        return jsonify({"error": "PRISM engine failed to initialise."}), 500

    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400

    symbol    = normalise_ticker(ticker)
    timeframe = request.args.get("timeframe", "").strip().upper()

    if timeframe:
        mapped = TIMEFRAME_MAP.get(timeframe)
        if not mapped:
            return jsonify({
                "error": f"Invalid timeframe. Supported: {', '.join(TIMEFRAME_MAP)}"
            }), 400
        period, interval = mapped
    else:
        period   = request.args.get("period",   "60d").strip()
        interval = request.args.get("interval", "1d").strip()

    try:
        report = prism.run_one_ticker(
            symbol, period=period, interval=interval,
            quiet=True, include_chart_history=True,
        )
        if not report:
            return jsonify({"error": f"Could not analyse {symbol}. Check ticker or network."}), 404

        # Attach cached LLM insights
        llm = load_cached_llm_insights(symbol)
        report["llm_insights"] = llm.get("insights") if llm else {}

        return jsonify(report)

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal error during analysis."}), 500


@app.route("/api/history/<ticker>")
def get_history(ticker):
    """
    Return lightweight OHLCV time-series directly from yfinance for chart rendering.
    Much faster than /api/analyze for chart-only refreshes.
    """
    symbol = normalise_ticker(ticker.strip().upper())
    period   = request.args.get("period",   "1y").strip()
    interval = request.args.get("interval", "1d").strip()

    try:
        stock = yf.Ticker(symbol)

        # Progressive interval fallback
        fallbacks = {
            "2m":  [("2m","5m","15m","30m")],
            "15m": ["15m","30m","60m"],
            "60m": ["60m","1d"],
        }
        intervals_to_try = fallbacks.get(interval, [interval])

        hist = None
        used_interval = interval
        for iv in intervals_to_try:
            try:
                df = stock.history(period=period, interval=iv, auto_adjust=False, actions=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    hist = df
                    used_interval = iv
                    break
            except Exception:
                pass
            try:
                df = yf.download(symbol, period=period, interval=iv,
                                 progress=False, auto_adjust=False,
                                 actions=False, threads=False)
                df = _normalise_download_frame(df, symbol)
                if df is not None and not df.empty and "Close" in df.columns:
                    hist = df
                    used_interval = iv
                    break
            except Exception:
                pass

        if hist is None or hist.empty or "Close" not in hist.columns:
            return jsonify({"symbol": symbol, "period": period, "interval": interval, "data": []})

        ts     = _index_to_unix(hist.index)
        close  = pd.to_numeric(hist["Close"],  errors="coerce").values
        open_  = pd.to_numeric(hist.get("Open",  hist["Close"]), errors="coerce").values
        high   = pd.to_numeric(hist.get("High",  hist["Close"]), errors="coerce").values
        low    = pd.to_numeric(hist.get("Low",   hist["Close"]), errors="coerce").values
        volume = pd.to_numeric(
            hist.get("Volume", pd.Series(0, index=hist.index)), errors="coerce"
        ).fillna(0).values

        mask = np.isfinite(close) & np.isfinite(open_) & np.isfinite(ts) & (ts > 0)
        data = [
            {"time": int(t), "open": float(o), "high": float(h),
             "low": float(l), "close": float(c), "volume": float(v)}
            for t, o, h, l, c, v in zip(
                ts[mask], open_[mask], high[mask], low[mask], close[mask], volume[mask]
            )
        ]
        return jsonify({"symbol": symbol, "period": period, "interval": used_interval, "data": data})

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal error fetching history."}), 500


@app.route("/api/llm-insights")
def get_llm_insights():
    """
    Return cached LLM insights for a ticker.
    If no cache exists and DEMO_MODE is false, generate live.
    """
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    symbol = normalise_ticker(ticker)
    cached = load_cached_llm_insights(symbol)
    if cached:
        return jsonify(cached)

    if not DEMO_MODE and prism:
        try:
            report = prism.run_one_ticker(symbol, quiet=True, include_chart_history=False)
            if report:
                insights = generate_llm_insights(report)
                return jsonify(insights)
        except Exception:
            pass

    return jsonify({"message": "No LLM insights available yet. Run the daily batch."}), 404


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Log user feedback to JSON file."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    item = dict(payload)
    item["timestamp"] = datetime.now(timezone.utc).isoformat()
    log_path = _feedback_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        logs = loaded if isinstance(loaded, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    logs.append(item)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except OSError:
        return jsonify({"error": "Failed to write feedback"}), 500

    return jsonify({"status": "success", "demo_mode": DEMO_MODE})


@app.route("/api/feedback/status")
def feedback_status():
    log_path = _feedback_log_path()
    count = 0
    last_ts = None
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if isinstance(logs, list):
                count = len(logs)
                if logs:
                    last_ts = logs[-1].get("timestamp")
        except Exception:
            pass
    return jsonify({
        "demo_mode": DEMO_MODE,
        "feedback_log_path": str(log_path),
        "feedback_log_entries": count,
        "last_timestamp": last_ts,
    })


@app.route("/api/admin/feedback")
def admin_feedback():
    log_path = _feedback_log_path()
    if log_path.exists():
        return send_file(str(log_path), mimetype="application/json")
    return jsonify({"status": "empty", "message": "No feedback logged yet."})


@app.route("/api/session-summary/latest")
def latest_session_summary():
    path = SESSIONS_DIR / "latest_session_summary.json"
    if not path.exists():
        return jsonify({"error": "No session summary found yet."}), 404
    return send_file(str(path), mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
