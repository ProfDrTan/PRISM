"""
PRISM — llm_insights.py
=======================
Multi-LLM insights engine for PRISM.

Queries four AI providers with identical structured prompts and returns
comparative forecasts for the dashboard's LLM Insights Panel.

Providers:
  1. Claude (Anthropic)   — claude-sonnet-4-20250514
  2. ChatGPT (OpenAI)     — gpt-4o-mini
  3. DeepSeek             — deepseek-chat
  4. Gemini (Google)      — gemini-2.5-flash

Design: reports are pre-generated in batch (run_daily.py) and cached as JSON.
Live API requests are only made during batch generation, not on user requests.

Author: Professor Dr. Teik Kheong Tan
Built with: Claude (Anthropic)
"""

import os
import json
import time
from pathlib import Path

# ── API clients (graceful import) ─────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import google.generativeai as _genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

try:
    from openai import OpenAI as _OpenAI_DS  # DeepSeek uses OpenAI-compatible API
    _HAS_DEEPSEEK = True
except ImportError:
    _HAS_DEEPSEEK = False


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or None
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")    or None
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY")    or None
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")  or None

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_REPORTS_DIR = PROJECT_ROOT / "data" / "LLM_reports"


def _build_prompt(symbol: str, label: str, price: float, trend_label: str,
                  trend_direction: str, sentiment_label: str, sentiment_score: float,
                  headlines: list[str], alert: bool, alert_message: str) -> str:
    """
    Build the shared structured prompt sent identically to all four LLMs.
    """
    headlines_str = "\n".join(f"  - {h}" for h in (headlines or [])[:8])
    alert_str = f"⚠ PRISM ALERT ACTIVE: {alert_message}" if alert else "✓ No divergence alert."

    return f"""You are a market intelligence synthesis assistant for PRISM, an educational
decision-support tool. Analyse the evidence below and produce a structured
weekly market regime recommendation.

INSTRUMENT: {label} ({symbol})
CURRENT PRICE: {price}
TREND PREDICTION (machine learning): {trend_label} ({trend_direction})
SENTIMENT (FinBERT NLP, {sentiment_score:+.2f}): {sentiment_label}
PRISM DIVERGENCE STATUS: {alert_str}

RECENT HEADLINES:
{headlines_str}

Respond in EXACTLY this structure (no extra text before or after):

1. REGIME: [Bullish / Bearish / Neutral / Uncertain]
2. CONFIDENCE: [Low / Medium / High]
3. SUPPORTING EVIDENCE: (3 bullet points max)
4. KEY CONTRADICTIONS: (2 bullet points max)
5. INVALIDATION: What single event would change this view?
6. PREDICTED MOVE: {symbol} — direction and % range for next session
7. PLAIN ENGLISH: 2–3 sentences a non-expert can understand
8. DISCLAIMER: One sentence reminding the reader this is not financial advice.
"""


def _query_claude(prompt: str) -> dict:
    if not _HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        return {"model": "claude", "status": "unavailable", "response": None}
    try:
        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text if message.content else ""
        return {"model": "claude", "status": "ok", "response": text}
    except Exception as e:
        return {"model": "claude", "status": "error", "error": str(e), "response": None}


def _query_chatgpt(prompt: str) -> dict:
    if not _HAS_OPENAI or not OPENAI_API_KEY:
        return {"model": "chatgpt", "status": "unavailable", "response": None}
    try:
        client = _OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content if resp.choices else ""
        return {"model": "chatgpt", "status": "ok", "response": text}
    except Exception as e:
        return {"model": "chatgpt", "status": "error", "error": str(e), "response": None}


def _query_deepseek(prompt: str) -> dict:
    if not _HAS_DEEPSEEK or not DEEPSEEK_API_KEY:
        return {"model": "deepseek", "status": "unavailable", "response": None}
    try:
        client = _OpenAI_DS(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content if resp.choices else ""
        return {"model": "deepseek", "status": "ok", "response": text}
    except Exception as e:
        return {"model": "deepseek", "status": "error", "error": str(e), "response": None}


def _query_gemini(prompt: str) -> dict:
    if not _HAS_GEMINI or not GOOGLE_API_KEY:
        return {"model": "gemini", "status": "unavailable", "response": None}
    try:
        _genai.configure(api_key=GOOGLE_API_KEY)
        model = _genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else ""
        return {"model": "gemini", "status": "ok", "response": text}
    except Exception as e:
        return {"model": "gemini", "status": "error", "error": str(e), "response": None}


def generate_llm_insights(report: dict) -> dict:
    """
    Query all four LLMs with identical prompts and return a structured
    insights dict suitable for the PRISM dashboard.

    Caches results to data/LLM_reports/<symbol>_llm.json.
    """
    meta      = report.get("meta", {})
    market    = report.get("market", {})
    sentiment = report.get("sentiment", {})
    signals   = report.get("signals", {})

    symbol    = meta.get("symbol", "UNKNOWN")
    label     = meta.get("label", symbol)
    price     = market.get("current_price", 0.0)
    trend_lbl = market.get("trend_label", "Unknown")
    trend_dir = market.get("trend_direction", "neutral")
    sent_lbl  = sentiment.get("label", "Neutral")
    sent_scr  = sentiment.get("score", 0.0)
    headlines = sentiment.get("headlines", [])
    alert     = signals.get("prism_alert", False)
    alert_msg = signals.get("alert_message", "")

    prompt = _build_prompt(
        symbol, label, price, trend_lbl, trend_dir,
        sent_lbl, sent_scr, headlines, alert, alert_msg,
    )

    results = {
        "claude":   _query_claude(prompt),
        "chatgpt":  _query_chatgpt(prompt),
        "deepseek": _query_deepseek(prompt),
        "gemini":   _query_gemini(prompt),
    }

    payload = {
        "meta": {
            "symbol": symbol,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_date": meta.get("market_session_date", ""),
        },
        "insights": results,
        "prompt_used": prompt,
    }

    # Cache to disk
    LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LLM_REPORTS_DIR / f"{symbol}_llm.json"
    existing = []
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing = data if isinstance(data, list) else [data]
        except Exception:
            pass
    existing.append(payload)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return payload


def load_cached_llm_insights(symbol: str) -> dict | None:
    """Load the most recent cached LLM insights for a given ticker."""
    cache_path = LLM_REPORTS_DIR / f"{symbol}_llm.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        for record in reversed(records):
            if record.get("meta", {}).get("symbol", "").upper() == symbol.upper():
                return record
    except Exception:
        pass
    return None
