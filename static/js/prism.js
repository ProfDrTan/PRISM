/* PRISM — prism.js
   Author: Professor Dr. Teik Kheong Tan
   Frontend dashboard logic for PRISM.
   ================================================ */

'use strict';

let _chart = null;
let _candleSeries = null;
let _volumeSeries = null;
let _currentTicker = 'AAPL';

// ── Entry point ───────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('tickerInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') runAnalysis();
  });
  initChart();
  runAnalysis();
});

function quickTicker(symbol) {
  document.getElementById('tickerInput').value = symbol;
  runAnalysis();
}

// ── Main analysis flow ────────────────────────────────────────────────────────

async function runAnalysis() {
  const ticker    = document.getElementById('tickerInput').value.trim().toUpperCase();
  const timeframe = document.getElementById('timeframeSelect').value;
  if (!ticker) return;

  _currentTicker = ticker;
  showLoading(`Running PRISM analysis for ${ticker}…`);

  try {
    const res  = await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}&timeframe=${timeframe}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Analysis failed. Check ticker symbol.');
      return;
    }

    renderDashboard(data);
    showDashboard();

    // Render chart from analysis data (chart_history already included)
    if (data.chart_history && data.chart_history.length > 0) {
      updateChart(data.chart_history, ticker);
    } else {
      fetchAndRenderChart(ticker, timeframe);
    }

    // LLM insights
    renderLLMInsights(data.llm_insights || {});

  } catch (err) {
    showError('Network error. Please try again.');
    console.error(err);
  }
}

// ── Render helpers ─────────────────────────────────────────────────────────────

function renderDashboard(data) {
  const market    = data.market    || {};
  const sentiment = data.sentiment || {};
  const signals   = data.signals   || {};
  const meta      = data.meta      || {};

  // ── PRISM Alert ──────────────────────────────────────────────
  const alertCard  = document.getElementById('alertCard');
  const alertLight = document.getElementById('alertLight');
  const alertLevel = document.getElementById('alertLevel');
  const alertMsg   = document.getElementById('alertMsg');

  const level = (signals.alert_level || 'CLEAR').toUpperCase();
  alertLevel.textContent = level;
  alertLevel.className   = 'alert-level level-' + level.toLowerCase();
  alertMsg.textContent   = signals.alert_message || 'Signals aligned';

  if (level === 'HIGH') {
    alertLight.textContent = '🔴';
    alertCard.classList.add('alert-active');
    alertCard.classList.remove('alert-clear');
  } else if (level === 'MEDIUM') {
    alertLight.textContent = '🟡';
    alertCard.classList.add('alert-active');
    alertCard.classList.remove('alert-clear');
  } else {
    alertLight.textContent = '🟢';
    alertCard.classList.remove('alert-active');
    alertCard.classList.add('alert-clear');
  }

  // ── Price ────────────────────────────────────────────────────
  const currentPrice   = market.current_price;
  const predictedPrice = market.predicted_price_next_session;
  const trendLabel     = market.trend_label || '—';
  const trendDir       = market.trend_direction || 'neutral';

  document.getElementById('currentPrice').textContent =
    currentPrice != null ? formatPrice(currentPrice) : '—';
  document.getElementById('predictedPrice').textContent =
    predictedPrice != null ? `→ ${formatPrice(predictedPrice)}` : '→ —';

  const badge = document.getElementById('trendBadge');
  badge.textContent = trendLabel;
  badge.className   = 'trend-badge ' + trendDir;

  // ── Sentiment ────────────────────────────────────────────────
  const score     = sentiment.score  || 0;
  const sentLabel = sentiment.label  || 'Neutral';
  const hCount    = sentiment.headline_count || 0;
  const posCount  = sentiment.positive_count || 0;
  const negCount  = sentiment.negative_count || 0;

  const scoreEl = document.getElementById('sentimentScore');
  scoreEl.textContent = (score >= 0 ? '+' : '') + score.toFixed(2);
  scoreEl.className   = 'sentiment-score ' + sentimentClass(sentLabel);

  document.getElementById('sentimentLabel').textContent = sentLabel;

  // Sentiment bar: map -1..+1 to 0..100%
  const pct = Math.round((score + 1) / 2 * 100);
  const bar = document.getElementById('sentimentBar');
  bar.style.width = pct + '%';
  bar.style.background = score > 0.15 ? '#16a34a' : score < -0.15 ? '#dc2626' : '#94a3b8';

  document.getElementById('sentimentStats').textContent =
    `${hCount} headlines · ${posCount}↑ ${negCount}↓`;

  // ── EMA Zone ─────────────────────────────────────────────────
  document.getElementById('emaZone').textContent    = signals.ema_zone || '—';
  document.getElementById('emaVals').textContent    =
    `EMA 8: ${signals.ema_8 != null ? formatPrice(signals.ema_8) : '—'} · EMA 21: ${signals.ema_21 != null ? formatPrice(signals.ema_21) : '—'}`;

  const crossEl = document.getElementById('emaCrossSignal');
  crossEl.textContent = signals.ema_cross_signal || '—';
  crossEl.className   = 'ema-signal ' + ((signals.ema_cross_signal || '').toLowerCase());

  // ── Chart label ──────────────────────────────────────────────
  document.getElementById('chartLabel').textContent =
    `CANDLESTICK CHART — ${meta.label || meta.symbol || _currentTicker}`;

  // ── Headlines ────────────────────────────────────────────────
  const headlines = (data.sentiment || {}).headlines || [];
  renderHeadlines(headlines);
}

function renderHeadlines(headlines) {
  const list = document.getElementById('headlinesList');
  if (!headlines.length) {
    list.innerHTML = '<p class="placeholder">No headlines available.</p>';
    return;
  }
  list.innerHTML = headlines.map(h => {
    const dot = '<span class="headline-dot neu"></span>';
    return `<div class="headline-item">${dot}<span>${escapeHtml(h)}</span></div>`;
  }).join('');
}

function renderLLMInsights(insights) {
  const grid = document.getElementById('llmGrid');
  const models = [
    { key: 'claude',   name: '🟠 Claude (Anthropic)',  cls: 'claude'   },
    { key: 'chatgpt',  name: '🟢 ChatGPT (OpenAI)',    cls: 'chatgpt'  },
    { key: 'deepseek', name: '🔵 DeepSeek',            cls: 'deepseek' },
    { key: 'gemini',   name: '🟣 Gemini (Google)',     cls: 'gemini'   },
  ];

  if (!insights || !Object.keys(insights).length) {
    grid.innerHTML = '<div class="llm-loading">LLM insights not yet available for this ticker.</div>';
    return;
  }

  grid.innerHTML = models.map(m => {
    const data   = insights[m.key] || {};
    const status = data.status || 'unavailable';
    const resp   = data.response || '';

    if (status !== 'ok' || !resp) {
      return `<div class="llm-card">
        <div class="llm-name ${m.cls}">${m.name}</div>
        <div class="llm-unavailable">Insights unavailable.<br>Configure API key to enable.</div>
      </div>`;
    }

    // Parse regime and confidence from structured response
    const regime     = extractField(resp, '1. REGIME')     || 'Uncertain';
    const confidence = extractField(resp, '2. CONFIDENCE') || '—';
    const plain      = extractField(resp, '7. PLAIN ENGLISH') || resp.slice(0, 180);

    const regimeClass = regime.toLowerCase().replace(/[^a-z]/g, '');

    return `<div class="llm-card">
      <div class="llm-name ${m.cls}">${m.name}</div>
      <div class="llm-regime ${regimeClass}">${regime}</div>
      <div class="llm-confidence">Confidence: ${confidence}</div>
      <div class="llm-summary">${escapeHtml(plain.slice(0, 200))}</div>
    </div>`;
  }).join('');
}

// ── TradingView Chart ─────────────────────────────────────────────────────────

function initChart() {
  const container = document.getElementById('chart');
  _chart = LightweightCharts.createChart(container, {
    width: container.offsetWidth,
    height: 400,
    layout: {
      background: { color: '#ffffff' },
      textColor:  '#475569',
    },
    grid: {
      vertLines:  { color: '#f1f5f9' },
      horzLines:  { color: '#f1f5f9' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#e2e8f0' },
    timeScale: { borderColor: '#e2e8f0', timeVisible: true },
  });

  _candleSeries = _chart.addCandlestickSeries({
    upColor:   '#16a34a',
    downColor: '#dc2626',
    borderUpColor:   '#16a34a',
    borderDownColor: '#dc2626',
    wickUpColor:     '#16a34a',
    wickDownColor:   '#dc2626',
  });

  _volumeSeries = _chart.addHistogramSeries({
    color:       '#3b82f6',
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  window.addEventListener('resize', () => {
    _chart.applyOptions({ width: container.offsetWidth });
  });
}

function updateChart(data, ticker) {
  if (!_candleSeries || !data || !data.length) return;

  const candles = data
    .filter(d => d.time && isFinite(d.close) && isFinite(d.open))
    .sort((a, b) => a.time - b.time)
    .map(d => ({
      time:  d.time,
      open:  d.open,
      high:  d.high,
      low:   d.low,
      close: d.close,
    }));

  const volumes = data
    .filter(d => d.time && isFinite(d.volume))
    .sort((a, b) => a.time - b.time)
    .map(d => ({
      time:  d.time,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(22,163,74,0.3)' : 'rgba(220,38,38,0.3)',
    }));

  _candleSeries.setData(candles);
  _volumeSeries.setData(volumes);
  _chart.timeScale().fitContent();
}

async function fetchAndRenderChart(ticker, timeframe) {
  const tfMap = {
    '1D': {period:'1d',  interval:'2m'},
    '5D': {period:'5d',  interval:'15m'},
    '1M': {period:'1mo', interval:'1h'},
    '6M': {period:'6mo', interval:'1d'},
    'YTD':{period:'ytd', interval:'1d'},
    '1Y': {period:'1y',  interval:'1d'},
    '5Y': {period:'5y',  interval:'1wk'},
  };
  const tf = tfMap[timeframe] || {period:'1mo', interval:'1d'};
  try {
    const res  = await fetch(`/api/history/${encodeURIComponent(ticker)}?period=${tf.period}&interval=${tf.interval}`);
    const data = await res.json();
    if (data.data && data.data.length) {
      updateChart(data.data, ticker);
    }
  } catch (e) {
    console.warn('Chart fetch failed:', e);
  }
}

// ── UI state helpers ──────────────────────────────────────────────────────────

function showLoading(msg) {
  hide('dashboard');
  hide('errorState');
  show('loadingState');
  document.getElementById('loadingMsg').textContent = msg || 'Loading…';
  document.getElementById('analyseBtn').disabled = true;
}

function showDashboard() {
  hide('loadingState');
  hide('errorState');
  show('dashboard');
  document.getElementById('analyseBtn').disabled = false;
}

function showError(msg) {
  hide('loadingState');
  hide('dashboard');
  show('errorState');
  document.getElementById('errorMsg').textContent = msg;
  document.getElementById('analyseBtn').disabled = false;
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

// ── Utility ───────────────────────────────────────────────────────────────────

function formatPrice(n) {
  if (n == null || !isFinite(n)) return '—';
  if (n > 1000) return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  return '$' + n.toFixed(2);
}

function sentimentClass(label) {
  if (!label) return 'neu';
  const l = label.toLowerCase();
  if (l === 'positive') return 'pos';
  if (l === 'negative') return 'neg';
  return 'neu';
}

function extractField(text, fieldName) {
  if (!text || !fieldName) return '';
  const pattern = new RegExp(fieldName + '[:\\s]+([^\\n]+)', 'i');
  const match   = text.match(pattern);
  return match ? match[1].trim().replace(/^\[|\]$/g, '') : '';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}
