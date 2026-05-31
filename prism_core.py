"""
PRISM — prism_core.py
=====================
Core intelligence engine for PRISM (Predictive Reasoning and Intelligence
Synthesis for Markets).

Architecture:
  - PRISMSystem class: central object; initialised once at app startup
  - Market data: yfinance (OHLCV) with multi-level fallback
  - Sentiment: FinBERT (ProsusAI/finbert) via HuggingFace Transformers
  - Trend prediction: Ridge regression + Random Forest ensemble
  - News: NewsAPI + Webz.io + Google News RSS with deduplication
  - Divergence detection: Check Engine Light logic
  - Report storage: accumulative JSON per ticker

Author: Professor Dr. Teik Kheong Tan
Built with: Claude (Anthropic)
"""

import os
import re
import json
import time
import argparse
import requests
import feedparser

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import numpy as np
import pandas as pd
import yfinance as yf

from pathlib import Path
from newsapi import NewsApiClient
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from storage_paths import resolve_data_dir

# ── Optional matplotlib (graceful fallback) ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# ── Environment ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEWS_API_KEY   = os.environ.get("NEWS_API_KEY") or None
NEWS_API_KEYS  = [k.strip() for k in os.environ.get("NEWS_API_KEYS", "").split(",") if k.strip()]
WEBZ_API_KEY   = os.environ.get("WEBZ_API_KEY") or None
DEMO_MODE      = os.environ.get("DEMO_MODE", "false").lower() == "true"
PROJECT_ROOT   = Path(__file__).resolve().parent
DATA_DIR       = resolve_data_dir(PROJECT_ROOT, DEMO_MODE)
SESSIONS_DIR   = DATA_DIR / "sessions"
CHARTS_DIR     = DATA_DIR / "charts"
YF_CACHE_DIR   = DATA_DIR / "yfinance_tz_cache"

# ── Ticker configuration ───────────────────────────────────────────────────────
TICKER_ALIASES = {
    "GOOGL": "GOOG",
    "SPY":   "SPX",
    "QQQ":   "NDX",
}

COMPANY_NAME_TO_TICKERS = {
    "APPLE":      ["AAPL"],
    "MICROSOFT":  ["MSFT"],
    "NVIDIA":     ["NVDA"],
    "TESLA":      ["TSLA"],
    "META":       ["META"],
    "AMAZON":     ["AMZN"],
    "GOOGLE":     ["GOOG", "GOOGL"],
    "ALPHABET":   ["GOOG", "GOOGL"],
    "AMD":        ["AMD"],
    "SP500":      ["^GSPC"],
    "S&P500":     ["^GSPC"],
    "NASDAQ":     ["^NDX"],
    "RUSSELL":    ["^RUT"],
}

# Default watchlist: US Tech + core indices
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOG", "AMD",  # US Tech
    "^GSPC", "^NDX", "^RUT",                                          # SPX, NDX, IWM
]

# Human-readable ticker labels for the dashboard
TICKER_LABELS = {
    "^GSPC": "S&P 500 (SPX)",
    "^NDX":  "Nasdaq 100 (NDX)",
    "^RUT":  "Russell 2000 (IWM)",
    "AAPL":  "Apple (AAPL)",
    "MSFT":  "Microsoft (MSFT)",
    "NVDA":  "NVIDIA (NVDA)",
    "TSLA":  "Tesla (TSLA)",
    "META":  "Meta (META)",
    "AMZN":  "Amazon (AMZN)",
    "GOOG":  "Alphabet (GOOG)",
    "AMD":   "AMD (AMD)",
}

# News search keywords per ticker
TICKER_NEWS_KEYWORDS = {
    "^GSPC": ["S&P 500", "SPX", "stock market"],
    "^NDX":  ["Nasdaq", "NDX", "tech stocks"],
    "^RUT":  ["Russell 2000", "small cap", "IWM"],
    "AAPL":  ["Apple", "AAPL", "iPhone", "Tim Cook"],
    "MSFT":  ["Microsoft", "MSFT", "Azure", "Satya Nadella"],
    "NVDA":  ["NVIDIA", "NVDA", "GPU", "Jensen Huang"],
    "TSLA":  ["Tesla", "TSLA", "Elon Musk", "electric vehicle"],
    "META":  ["Meta", "META", "Facebook", "Mark Zuckerberg"],
    "AMZN":  ["Amazon", "AMZN", "AWS", "Andy Jassy"],
    "GOOG":  ["Google", "Alphabet", "GOOG", "Sundar Pichai"],
    "AMD":   ["AMD", "Advanced Micro Devices", "Lisa Su"],
}

# ── URL deduplication helpers ──────────────────────────────────────────────────
_URL_NOISE = re.compile(r"[?#].*$")
_WORD_BOUNDARY = re.compile(r"\b")

def _normalise_url(url: str) -> str:
    return _URL_NOISE.sub("", str(url or "").strip().lower())


def _ticker_boundary_pattern(ticker: str) -> re.Pattern:
    """Compile a word-boundary regex for a ticker symbol."""
    escaped = re.escape(ticker.lstrip("^"))
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


# ── Ticker normalisation ───────────────────────────────────────────────────────
def normalise_ticker(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    return TICKER_ALIASES.get(token, token)


def sanitise_company_token(value: str) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


# ══════════════════════════════════════════════════════════════════════════════
#  PRISMSystem — the central intelligence class
# ══════════════════════════════════════════════════════════════════════════════
class PRISMSystem:
    """
    PRISM core intelligence engine.

    Responsibilities:
      - Market data acquisition and caching (yfinance)
      - Financial sentiment analysis (FinBERT)
      - Price trend prediction (Ridge + RandomForest ensemble)
      - Multi-source news aggregation with deduplication
      - Divergence detection (the PRISM Alert / Check Engine Light)
      - Accumulative report storage (JSON per ticker)
      - Session comparison (current vs previous session)
    """

    def __init__(self):
        print("\n📡 Initialising PRISM Intelligence Engine...")
        self._setup_cache_dirs()
        self._load_finbert()
        self._connect_news_apis()
        self.merge_alias_reports()
        print("📡 PRISM Ready.\n")

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _setup_cache_dirs(self):
        """Create required directories and configure yfinance cache."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            cache_mod = getattr(yf, "cache", None)
            setter = getattr(cache_mod, "set_cache_location", None)
            if callable(setter):
                setter(str(YF_CACHE_DIR))
            if hasattr(yf, "set_tz_cache_location"):
                yf.set_tz_cache_location(str(YF_CACHE_DIR))
        except Exception:
            pass
        self._probe_sqlite_cache()

    def _probe_sqlite_cache(self):
        """Test whether SQLite cache is writable; fall back to dummy objects if not."""
        try:
            import sqlite3
            probe = YF_CACHE_DIR / ".cache_probe.sqlite3"
            conn = sqlite3.connect(str(probe))
            conn.execute("CREATE TABLE IF NOT EXISTS _probe (id INTEGER)")
            conn.close()
            try:
                probe.unlink()
            except OSError:
                pass
        except Exception:
            try:
                cache_mod = getattr(yf, "cache", None)
                if cache_mod:
                    for mgr, dummy in [
                        ("_CookieCacheManager", "_CookieCacheDummy"),
                        ("_ISINCacheManager",   "_ISINCacheDummy"),
                        ("_TzCacheManager",     "_TzCacheDummy"),
                    ]:
                        if hasattr(cache_mod, mgr) and hasattr(cache_mod, dummy):
                            setattr(
                                getattr(cache_mod, mgr),
                                "_" + mgr.replace("Manager", "").lower() + "_cache",
                                getattr(cache_mod, dummy)(),
                            )
            except Exception:
                pass

    def _load_finbert(self):
        """Load ProsusAI/finbert for financial sentiment analysis."""
        print("  → Loading FinBERT (financial sentiment model)...")
        try:
            model_id = os.environ.get("PRISM_FINBERT_MODEL_ID", "ProsusAI/finbert")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, use_safetensors=False
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", model=model, tokenizer=tokenizer
            )
            print("  → FinBERT loaded ✓")
        except Exception as e:
            print(f"  → FinBERT failed to load: {e}. Falling back to lexicon scoring.")
            self.sentiment_analyzer = None

    def _connect_news_apis(self):
        """Establish NewsAPI and Webz.io connections."""
        self.news_api = None
        all_keys = list(NEWS_API_KEYS)
        if NEWS_API_KEY and NEWS_API_KEY not in all_keys:
            all_keys.append(NEWS_API_KEY)
        self._news_api_keys = all_keys

        if all_keys:
            self.news_api = NewsApiClient(api_key=all_keys[0])
            print(f"  → NewsAPI: {len(all_keys)} key(s) configured ✓")
        else:
            print("  → NewsAPI: no key found — simulation mode")

        self.webz_api_key = WEBZ_API_KEY
        if self.webz_api_key:
            print("  → Webz.io: configured ✓")
        else:
            print("  → Webz.io: not configured")

    # ── Market data ───────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame | None:
        """
        Fetch OHLCV data from yfinance with progressive interval fallback.
        Falls back to synthetic simulation data if all fetches fail.
        """
        stock = yf.Ticker(symbol)
        interval_fallbacks = {
            "2m":  ["2m", "5m", "15m", "30m"],
            "15m": ["15m", "30m", "60m"],
            "60m": ["60m", "1d"],
            "1h":  ["60m", "1d"],
        }
        attempts = [(period, iv) for iv in interval_fallbacks.get(interval, [interval])]
        if (period, interval) not in attempts:
            attempts.insert(0, (period, interval))

        for p, iv in attempts:
            try:
                df = stock.history(period=p, interval=iv, auto_adjust=False, actions=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    return df
            except Exception:
                pass
            try:
                df = yf.download(symbol, period=p, interval=iv,
                                 progress=False, auto_adjust=False,
                                 actions=False, threads=False)
                df = self._flatten_multiindex(df, symbol)
                if df is not None and not df.empty and "Close" in df.columns:
                    return df
            except Exception:
                pass

        print(f"  ⚠ Could not fetch live data for {symbol} — using simulation.")
        return self._simulate_ohlcv(symbol)

    def _flatten_multiindex(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if symbol in df.columns.get_level_values(-1):
                    return df.xs(symbol, axis=1, level=-1, drop_level=True)
            except Exception:
                pass
            df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        return df

    def _simulate_ohlcv(self, ticker: str, base_price: float | None = None) -> pd.DataFrame:
        """
        Generate deterministic synthetic OHLCV data for offline/demo use.
        Seeded from the ticker symbol for reproducibility.
        """
        symbol = str(ticker or "").strip().upper() or "DEMO"
        seed = sum((i + 1) * ord(c) for i, c in enumerate(symbol))
        rng = np.random.default_rng(seed)

        if base_price is None or not np.isfinite(base_price) or base_price <= 0:
            base_price = float(50 + (seed % 350))

        n = 60
        trend = np.linspace(-0.03, 0.03, n)
        noise = rng.normal(0.0, 0.01, n)
        close = np.maximum(1.0, base_price * (1.0 + trend + noise))
        close[-1] = max(1.0, float(base_price))
        open_ = close * (1.0 - rng.normal(0.0, 0.005, n))
        high  = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
        low   = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
        vol   = rng.integers(700_000, 3_500_000, size=n).astype(float)

        idx = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=n, freq="D", tz="UTC")
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
        df["returns_1d"] = df["Close"].pct_change().fillna(0.0)
        df["sma_5"]  = df["Close"].rolling(5,  min_periods=1).mean()
        df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()
        return df

    # ── Feature engineering ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as ML features."""
        df = df.copy()
        df["returns_1d"] = df["Close"].pct_change().fillna(0.0)
        df["returns_5d"] = df["Close"].pct_change(5).fillna(0.0)
        df["sma_5"]  = df["Close"].rolling(5,  min_periods=1).mean()
        df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["ema_8"]  = df["Close"].ewm(span=8,  adjust=False).mean()
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["volatility"] = df["returns_1d"].rolling(10, min_periods=2).std().fillna(0.0)
        df["volume_ma5"] = df["Volume"].rolling(5, min_periods=1).mean()
        df["price_vs_sma20"] = (df["Close"] / df["sma_20"] - 1.0).fillna(0.0)
        df["ema_cross"] = (df["ema_8"] - df["ema_21"]).fillna(0.0)
        return df

    # ── Trend prediction ──────────────────────────────────────────────────────

    def _predict_trend(self, df: pd.DataFrame) -> dict:
        """
        Predict next-session price using Ridge + RandomForest ensemble.
        Returns predicted price, trend label, and confidence score.
        """
        df = self._engineer_features(df)
        feature_cols = [
            "returns_1d", "returns_5d", "sma_5", "sma_10", "sma_20",
            "ema_8", "ema_21", "volatility", "volume_ma5",
            "price_vs_sma20", "ema_cross",
        ]
        available = [c for c in feature_cols if c in df.columns]
        df_clean = df[available + ["Close"]].dropna()

        if len(df_clean) < 10:
            current_price = float(df["Close"].iloc[-1])
            return {
                "predicted_price": current_price,
                "trend_label": "Insufficient Data",
                "trend_direction": "neutral",
                "confidence": 0.0,
            }

        X = df_clean[available].values[:-1]
        y = df_clean["Close"].values[1:]
        X_pred = df_clean[available].values[-1].reshape(1, -1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pred_scaled = scaler.transform(X_pred)

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        ridge_pred = float(ridge.predict(X_pred_scaled)[0])

        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        rf_pred = float(rf.predict(X_pred_scaled)[0])

        # Ensemble: weighted average (equal weight for now)
        predicted_price = (ridge_pred + rf_pred) / 2.0
        current_price = float(df_clean["Close"].iloc[-1])
        pct_change = (predicted_price - current_price) / current_price * 100

        if pct_change > 0.3:
            trend_label = "Uptrend"
            trend_direction = "up"
        elif pct_change < -0.3:
            trend_label = "Downtrend"
            trend_direction = "down"
        else:
            trend_label = "Sideways"
            trend_direction = "neutral"

        # Confidence: agreement between ridge and rf
        agreement = 1.0 - abs(ridge_pred - rf_pred) / (abs(current_price) + 1e-9)
        confidence = float(np.clip(agreement, 0.0, 1.0))

        return {
            "predicted_price": round(predicted_price, 4),
            "current_price": round(current_price, 4),
            "pct_change": round(pct_change, 4),
            "trend_label": trend_label,
            "trend_direction": trend_direction,
            "confidence": round(confidence, 4),
            "ridge_pred": round(ridge_pred, 4),
            "rf_pred": round(rf_pred, 4),
        }

    # ── Sentiment analysis ────────────────────────────────────────────────────

    def _score_headline(self, text: str) -> float:
        """
        Score a single headline with FinBERT.
        Returns: +1.0 (positive) to -1.0 (negative).
        Falls back to simple lexicon if FinBERT unavailable.
        """
        if not text or not text.strip():
            return 0.0

        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])[0]
                label = result["label"].lower()
                score = float(result["score"])
                if label == "positive":
                    return score
                elif label == "negative":
                    return -score
                else:
                    return 0.0
            except Exception:
                pass

        # Lexicon fallback
        positive_words = {"beat", "surge", "rally", "gain", "record", "strong", "growth",
                          "profit", "bull", "up", "rise", "high", "boost"}
        negative_words = {"miss", "fall", "drop", "loss", "weak", "crash", "bear",
                          "down", "decline", "cut", "risk", "concern", "sell"}
        tokens = set(text.lower().split())
        pos = len(tokens & positive_words)
        neg = len(tokens & negative_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _aggregate_sentiment(self, headlines: list[str]) -> dict:
        """Compute aggregate sentiment score and label from a list of headlines."""
        if not headlines:
            return {"score": 0.0, "label": "Neutral", "headline_count": 0}

        scores = [self._score_headline(h) for h in headlines]
        avg = float(np.mean(scores))

        if avg > 0.15:
            label = "Positive"
        elif avg < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "score": round(avg, 4),
            "label": label,
            "headline_count": len(scores),
            "positive_count": sum(1 for s in scores if s > 0.1),
            "negative_count": sum(1 for s in scores if s < -0.1),
            "neutral_count":  sum(1 for s in scores if -0.1 <= s <= 0.1),
        }

    # ── News fetching ─────────────────────────────────────────────────────────

    def _fetch_headlines_newsapi(self, symbol: str, keywords: list[str]) -> list[str]:
        """Fetch headlines from NewsAPI for a given ticker."""
        if not self.news_api:
            return []
        query = " OR ".join(f'"{kw}"' for kw in keywords[:3])
        headlines = []
        for key in self._news_api_keys:
            try:
                client = NewsApiClient(api_key=key)
                resp = client.get_everything(
                    q=query, language="en", sort_by="publishedAt",
                    page_size=20, page=1,
                )
                articles = resp.get("articles", [])
                headlines = [a.get("title", "") for a in articles if a.get("title")]
                if headlines:
                    break
            except Exception:
                continue
        return headlines

    def _fetch_headlines_gnews(self, symbol: str, keywords: list[str]) -> list[str]:
        """Fetch headlines from Google News RSS."""
        headlines = []
        for kw in keywords[:2]:
            try:
                url = f"https://news.google.com/rss/search?q={requests.utils.quote(kw)}+stock&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    title = getattr(entry, "title", "")
                    if title:
                        headlines.append(title)
            except Exception:
                pass
        return headlines

    def _fetch_headlines_webz(self, symbol: str, keywords: list[str]) -> list[str]:
        """Fetch headlines from Webz.io."""
        if not self.webz_api_key:
            return []
        query = " OR ".join(keywords[:3])
        try:
            resp = requests.get(
                "https://api.webz.io/newsApiLite",
                params={"token": self.webz_api_key, "q": query, "size": 20},
                timeout=10,
            )
            data = resp.json()
            return [p.get("title", "") for p in data.get("posts", []) if p.get("title")]
        except Exception:
            return []

    def _collect_headlines(self, symbol: str) -> list[str]:
        """
        Aggregate headlines from all configured sources.
        Deduplicates by normalised title.
        """
        keywords = TICKER_NEWS_KEYWORDS.get(symbol, [symbol.lstrip("^")])
        raw = []
        raw += self._fetch_headlines_newsapi(symbol, keywords)
        raw += self._fetch_headlines_gnews(symbol, keywords)
        raw += self._fetch_headlines_webz(symbol, keywords)

        # Deduplicate by lowercased title
        seen = set()
        deduped = []
        for h in raw:
            key = re.sub(r"\s+", " ", h.strip().lower())
            if key and key not in seen:
                seen.add(key)
                deduped.append(h.strip())

        return deduped[:40]  # cap at 40 per ticker

    # ── Divergence detection (PRISM Alert) ───────────────────────────────────

    def _compute_prism_alert(
        self, sentiment: dict, trend: dict
    ) -> dict:
        """
        The PRISM Alert — flags divergence between sentiment and price trend.

        Divergence occurs when:
          - Sentiment is Positive AND trend is Downtrend
          - Sentiment is Negative AND trend is Uptrend

        This is the 'Check Engine Light' — it does NOT say buy or sell.
        It says: signals disagree — pause and investigate.
        """
        sent_label  = sentiment.get("label", "Neutral")
        trend_dir   = trend.get("trend_direction", "neutral")
        sent_score  = sentiment.get("score", 0.0)

        alert_on = (
            (sent_label == "Positive" and trend_dir == "down") or
            (sent_label == "Negative" and trend_dir == "up")
        )

        if alert_on:
            if sent_label == "Positive" and trend_dir == "down":
                message = (
                    "PRISM ALERT: Headlines are positive but price trend is bearish. "
                    "Market may be overconfident. Sentiment and technicals diverge."
                )
            else:
                message = (
                    "PRISM ALERT: Headlines are negative but price trend is bullish. "
                    "Possible contrarian setup or news-driven overreaction. "
                    "Sentiment and technicals diverge."
                )
        else:
            message = "Signals aligned — no divergence detected."

        # Divergence magnitude: how far apart are the signals?
        trend_score = {"up": 1.0, "neutral": 0.0, "down": -1.0}.get(trend_dir, 0.0)
        divergence_magnitude = abs(sent_score - trend_score)

        return {
            "alert": alert_on,
            "alert_level": "HIGH" if (alert_on and divergence_magnitude > 0.7)
                           else "MEDIUM" if alert_on
                           else "CLEAR",
            "message": message,
            "divergence_magnitude": round(divergence_magnitude, 4),
            "sentiment_direction": sent_label,
            "trend_direction": trend_dir,
        }

    # ── EMA signal (for dashboard) ────────────────────────────────────────────

    def _compute_ema_signals(self, df: pd.DataFrame) -> dict:
        """
        Compute 8/21 EMA cross signal — mirrors the Technical Agent on the
        DT3 Market Intelligence website.
        """
        df = self._engineer_features(df)
        if len(df) < 21:
            return {"zone": "Insufficient Data", "ema_8": None, "ema_21": None}

        ema_8  = float(df["ema_8"].iloc[-1])
        ema_21 = float(df["ema_21"].iloc[-1])
        price  = float(df["Close"].iloc[-1])

        if ema_8 > ema_21 and price > ema_8:
            zone = "Zone 1 — Bullish"
            zone_code = 1
        elif ema_8 > ema_21 and price < ema_8:
            zone = "Zone 2 — Compression"
            zone_code = 2
        elif ema_8 < ema_21:
            zone = "Zone 3 — Bearish"
            zone_code = 3
        else:
            zone = "Zone 4 — Recovery"
            zone_code = 4

        return {
            "zone": zone,
            "zone_code": zone_code,
            "ema_8": round(ema_8, 4),
            "ema_21": round(ema_21, 4),
            "price": round(price, 4),
            "ema_cross_signal": "Bullish" if ema_8 > ema_21 else "Bearish",
        }

    # ── Main analysis entry point ─────────────────────────────────────────────

    def run_one_ticker(
        self,
        ticker: str,
        period: str = "60d",
        interval: str = "1d",
        quiet: bool = False,
        include_chart_history: bool = True,
    ) -> dict | None:
        """
        Run a full PRISM analysis for a single ticker.

        Returns a report dict containing:
          - meta: timestamp, symbol, session date
          - market: price, prediction, OHLCV summary
          - sentiment: FinBERT scores and headline list
          - signals: PRISM alert, EMA zone, trend label
          - chart_history: OHLCV time-series for TradingView (if requested)
        """
        symbol = normalise_ticker(ticker)
        if not quiet:
            print(f"\n📡 Analysing: {symbol}")

        session_date = time.strftime("%Y-%m-%d")
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 1. Fetch market data
        df = self._fetch_ohlcv(symbol, period=period, interval=interval)
        if df is None or df.empty:
            return None

        # 2. Engineer features + predict trend
        trend = self._predict_trend(df)

        # 3. Collect headlines + compute sentiment
        headlines = self._collect_headlines(symbol)
        sentiment = self._aggregate_sentiment(headlines)

        # 4. PRISM alert (divergence detection)
        prism_alert = self._compute_prism_alert(sentiment, trend)

        # 5. EMA signals
        ema_signals = self._compute_ema_signals(df)

        # 6. Chart history for TradingView
        chart_history = []
        if include_chart_history:
            chart_history = self._build_chart_history(df)

        # 7. Assemble report
        report = {
            "meta": {
                "symbol": symbol,
                "label": TICKER_LABELS.get(symbol, symbol),
                "market_session_date": session_date,
                "generated_at": generated_at,
                "data_period": period,
                "data_interval": interval,
            },
            "market": {
                "current_price": trend.get("current_price"),
                "predicted_price_next_session": trend.get("predicted_price"),
                "pct_change_predicted": trend.get("pct_change"),
                "trend_label": trend.get("trend_label"),
                "trend_direction": trend.get("trend_direction"),
                "trend_confidence": trend.get("confidence"),
            },
            "sentiment": {
                "score": sentiment.get("score"),
                "label": sentiment.get("label"),
                "headline_count": sentiment.get("headline_count"),
                "positive_count": sentiment.get("positive_count"),
                "negative_count": sentiment.get("negative_count"),
                "neutral_count": sentiment.get("neutral_count"),
                "headlines": headlines[:10],  # top 10 for dashboard display
            },
            "signals": {
                "prism_alert": prism_alert.get("alert"),
                "alert_level": prism_alert.get("alert_level"),
                "alert_message": prism_alert.get("message"),
                "divergence_magnitude": prism_alert.get("divergence_magnitude"),
                "ema_zone": ema_signals.get("zone"),
                "ema_zone_code": ema_signals.get("zone_code"),
                "ema_8": ema_signals.get("ema_8"),
                "ema_21": ema_signals.get("ema_21"),
                "ema_cross_signal": ema_signals.get("ema_cross_signal"),
            },
            "chart_history": chart_history,
        }

        # Save report
        self.save_report(report, f"{symbol}_report.json")
        if not quiet:
            alert_str = "🔴 ALERT" if prism_alert["alert"] else "🟢 Clear"
            print(f"  Price: {trend.get('current_price')} → {trend.get('predicted_price')} "
                  f"({trend.get('trend_label')}) | Sentiment: {sentiment.get('label')} "
                  f"| {alert_str}")

        return report

    def _build_chart_history(self, df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to TradingView-compatible OHLCV list."""
        if df is None or df.empty:
            return []

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
        else:
            idx = pd.to_datetime(idx, utc=True, errors="coerce")

        ts = np.asarray(idx.asi8, dtype=np.int64) // 10**9

        close  = pd.to_numeric(df["Close"],  errors="coerce").values
        open_  = pd.to_numeric(df.get("Open",  df["Close"]), errors="coerce").values
        high   = pd.to_numeric(df.get("High",  df["Close"]), errors="coerce").values
        low    = pd.to_numeric(df.get("Low",   df["Close"]), errors="coerce").values
        volume = pd.to_numeric(df.get("Volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0).values

        mask = np.isfinite(close) & np.isfinite(open_) & np.isfinite(ts) & (ts > 0)
        return [
            {"time": int(t), "open": float(o), "high": float(h),
             "low": float(l), "close": float(c), "volume": float(v)}
            for t, o, h, l, c, v in zip(
                ts[mask], open_[mask], high[mask], low[mask], close[mask], volume[mask]
            )
        ]

    # ── Report persistence ────────────────────────────────────────────────────

    def save_report(self, report: dict, filename: str):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / filename
        reports = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports = data if isinstance(data, list) else [data]
            except Exception:
                pass
        reports.append(report)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)

    def _read_report_file(self, path: Path) -> list:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def _extract_session_date(self, report: dict) -> str | None:
        meta = report.get("meta", {})
        date = str(meta.get("market_session_date", "")).strip()
        if len(date) == 10:
            return date
        gen = str(meta.get("generated_at", "")).strip()
        if len(gen) >= 10:
            return gen[:10]
        return None

    def merge_alias_reports(self):
        """Merge aliased ticker reports into canonical files."""
        if not DATA_DIR.exists():
            return
        for alias, canonical in TICKER_ALIASES.items():
            alias_path     = DATA_DIR / f"{alias}_report.json"
            canonical_path = DATA_DIR / f"{canonical}_report.json"
            alias_reports  = self._read_report_file(alias_path)
            if not alias_reports:
                continue
            canonical_reports = self._read_report_file(canonical_path)
            merged = []
            for r in canonical_reports + alias_reports:
                if isinstance(r, dict):
                    r.setdefault("meta", {})["symbol"] = canonical
                    merged.append(r)
            if merged:
                with open(canonical_path, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)
                try:
                    alias_path.unlink()
                except OSError:
                    pass

    def save_session_summary(self, reports: list) -> str | None:
        """Save a cross-ticker session summary with session-over-session comparisons."""
        if not reports:
            return None
        session_dates = [self._extract_session_date(r) for r in reports if self._extract_session_date(r)]
        session_date  = max(session_dates) if session_dates else time.strftime("%Y-%m-%d")
        generated_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        latest = {}
        for r in reports:
            sym = str(r.get("meta", {}).get("symbol", "")).upper()
            if sym:
                latest[sym] = r

        payload = {
            "meta": {
                "generated_at": generated_at,
                "session_date": session_date,
                "symbols": sorted(latest.keys()),
                "report_count": len(latest),
            },
            "reports": [latest[s] for s in sorted(latest)],
        }

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        day_dir = SESSIONS_DIR / session_date
        day_dir.mkdir(parents=True, exist_ok=True)
        summary_path = day_dir / "session_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        latest_path = SESSIONS_DIR / "latest_session_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(summary_path)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Core — run analysis from CLI")
    parser.add_argument("--tickers", default="AAPL,MSFT,^GSPC", help="Comma-separated tickers")
    parser.add_argument("--period",  default="60d")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()

    system = PRISMSystem()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    reports = []
    for ticker in tickers:
        report = system.run_one_ticker(ticker, period=args.period, interval=args.interval)
        if report:
            reports.append(report)
    if reports:
        system.save_session_summary(reports)
    print("\n✅ PRISM analysis complete.")
