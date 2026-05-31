---
title: PRISM
emoji: 📡
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
pinned: true
---

# 📡 PRISM — Predictive Reasoning and Intelligence Synthesis for Markets

**Author: Professor Dr. Teik Kheong Tan**
CP3405 Design Thinking 3 · Singapore · 2026

---

## What PRISM Is

PRISM is an AI-powered market intelligence and decision support tool built for
teaching and research. It does **not** tell you to buy or sell. It is a
**"Check Engine Light"** — it flags when market signals disagree, so that
traders and students can pause, investigate, and reason carefully before acting.

> *"Don't predict the market. Understand it."*

---

## Core Features

- **Sentiment Analysis** — FinBERT (financial-domain NLP) analyses recent news headlines
- **Trend Engine** — Ridge regression + Random Forest on OHLCV data for next-session price prediction
- **Divergence Detection** — Flags when sentiment and price trend point in opposite directions
- **Multi-LLM Insights Panel** — Comparative forecasts from Claude (Anthropic), ChatGPT (OpenAI), DeepSeek, and Gemini
- **Interactive Dashboard** — TradingView Lightweight Charts with candlestick + volume overlays
- **Market Coverage** — US Tech (AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOG, AMD) + Indices (SPX, NDX, IWM)
- **Teaching Mode** — Annotated explanations of every signal for classroom use

---

## The PRISM Signal Framework

```
News Headlines  →  FinBERT Sentiment  ─┐
                                        ├──→  DIVERGENCE CHECK  →  🔴 PRISM Alert
OHLCV Data      →  Trend Prediction   ─┘
                                        
                    ↓
        Claude · ChatGPT · DeepSeek · Gemini
                    ↓
              LLM Insights Panel
                    ↓
           Human Score (your judgment)
```

---

## Scrum Team (CP3405 Model Solution)

| Role | Responsibility |
|------|---------------|
| Product Owner | Signal priority and feature roadmap |
| Scrum Master | Sprint cadence and retrospectives |
| AI Engineer | FinBERT pipeline and LLM integration |
| Data Scientist | Trend model and backtesting |
| Full-Stack Engineer | Flask API and dashboard |
| UI/UX Designer | Dashboard design and teaching annotations |

---

## Environment Variables (Hugging Face Secrets)

| Secret | Purpose |
|--------|---------|
| `NEWS_API_KEY` | NewsAPI primary key |
| `NEWS_API_KEYS` | Comma-separated fallback keys |
| `WEBZ_API_KEY` | Webz.io news source key |
| `OPENAI_API_KEY` | ChatGPT LLM insights |
| `ANTHROPIC_API_KEY` | Claude LLM insights |
| `GOOGLE_API_KEY` | Gemini LLM insights |
| `DEEPSEEK_API_KEY` | DeepSeek LLM insights |
| `DEMO_MODE` | Set `true` for classroom/public demo |

---

## Tech Stack

- **Backend:** Python 3.10 · Flask · Gunicorn
- **AI/ML:** FinBERT (ProsusAI) · scikit-learn · PyTorch · Transformers
- **Market Data:** yfinance · NewsAPI · Webz.io · Google News RSS
- **LLMs:** Claude (claude-sonnet-4-20250514) · GPT-4o-mini · DeepSeek V3 · Gemini 2.5 Flash
- **Frontend:** TradingView Lightweight Charts · Vanilla JS · CSS3
- **Deployment:** Docker · Hugging Face Spaces (pinned) · GitHub

---

## Deployment

Live demo: [https://huggingface.co/spaces/ProfDrTan/PRISM](https://huggingface.co/spaces/ProfDrTan/PRISM)

Source: [https://github.com/ProfDrTan/PRISM](https://github.com/ProfDrTan/PRISM)

---

## Disclaimer

PRISM provides decision support signals based on historical data and sentiment
analysis. It does **not** provide financial advice. All outputs are for
educational purposes only.

---

*Built with Claude (Anthropic) · Professor Dr. Teik Kheong Tan · 2026*
