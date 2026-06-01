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
    { key: 'grok',     name: '⚫ Grok (xAI)',            cls: 'grok'     },
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

    const regime     = extractField(resp, '1. REGIME')        || extractField(resp, 'REGIME')     || 'Uncertain';
    const confidence = extractField(resp, '2. CONFIDENCE')    || extractField(resp, 'CONFIDENCE') || '—';
    const plain      = extractField(resp, '7. PLAIN ENGLISH') || extractField(resp, 'PLAIN ENGLISH') || '';
    const predicted  = extractField(resp, '6. PREDICTED MOVE')|| extractField(resp, 'PREDICTED MOVE') || '';
    const summary    = plain || predicted || resp.slice(0, 220);

    const regimeClass = regime.toLowerCase().replace(/[^a-z]/g, '');
    const regimeColor = regime.toLowerCase().includes('bull') ? '#16a34a'
                      : regime.toLowerCase().includes('bear') ? '#dc2626'
                      : '#94a3b8';

    return `<div class="llm-card">
      <div class="llm-name ${m.cls}">${m.name}</div>
      <div class="llm-regime" style="color:${regimeColor};font-weight:700;font-size:1rem;margin:4px 0;">${regime}</div>
      <div class="llm-confidence" style="font-size:0.8rem;color:#64748b;margin-bottom:6px;">Confidence: ${confidence}</div>
      <div class="llm-summary" style="font-size:0.82rem;line-height:1.45;color:#334155;">${escapeHtml(summary.slice(0, 240))}</div>
    </div>`;
  }).join('');
}


// ── TradingView Chart ─────────────────────────────────────────────────────────

function initChart() {
  const container = document.getElementById('chart');
  const w = container.offsetWidth || 800;

  _chart = LightweightCharts.createChart(container, {
    width:  w,
    height: 500,
    layout: {
      background: { type: 'solid', color: '#ffffff' },
      textColor:  '#475569',
    },
    grid: {
      vertLines: { color: '#f1f5f9' },
      horzLines: { color: '#f1f5f9' },
    },
    crosshair:       { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#e2e8f0', scaleMargins: { top: 0.05, bottom: 0.15 } },
    timeScale:       { borderColor: '#e2e8f0', timeVisible: true },
  });

  // Candlestick series — exact IRIS colour scheme
  _candleSeries = _chart.addCandlestickSeries({
    upColor:         '#26a69a',
    downColor:       '#ef5350',
    borderVisible:   false,
    wickUpColor:     '#26a69a',
    wickDownColor:   '#ef5350',
  });

  // Volume histogram — exact IRIS pattern
  _volumeSeries = _chart.addHistogramSeries({
    color:        '#26a69a',
    priceFormat:  { type: 'volume' },
    priceScaleId: 'volume_scale',
  });

  // Critical: call applyOptions on the SERIES priceScale — NOT the chart
  _volumeSeries.priceScale().applyOptions({
    scaleMargins: { top: 0.82, bottom: 0 },
    drawTicks:    false,
    borderVisible: false,
    visible:      false,
  });

  window.addEventListener('resize', () => {
    _chart.applyOptions({ width: container.offsetWidth });
  });
}


// Convert a Unix timestamp (seconds) to a YYYY-MM-DD string in UTC
function unixToDateStr(ts) {
  const d = new Date(ts * 1000);
  const yyyy = d.getUTCFullYear();
  const mm   = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd   = String(d.getUTCDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function updateChart(rawData, ticker) {
  if (!_candleSeries || !rawData || !rawData.length) return;

  // Normalise timestamps — exact IRIS pattern
  const normalised = [];
  rawData.forEach(d => {
    if (!d || typeof d !== 'object') return;
    let t = d.time;
    if (typeof t === 'number' && isFinite(t)) {
      if (Math.abs(t) >= 1e12) t = Math.round(t / 1000); // ms -> s
      else if (Math.abs(t) < 1e8) return; // invalid
    } else if (typeof t === 'string') {
      const n = Number(t);
      if (!isNaN(n) && Math.abs(n) >= 1e8) {
        t = Math.abs(n) >= 1e12 ? Math.round(n / 1000) : Math.round(n);
      }
    }
    if (!t) return;
    const close = Number(d.close ?? d.value);
    const open  = Number(d.open  ?? close);
    const high  = Number(d.high  ?? close);
    const low   = Number(d.low   ?? close);
    if (!isFinite(close)) return;
    normalised.push({ time: t, open, high, low, close, volume: Number(d.volume) || 0 });
  });

  // Deduplicate + sort ascending
  const deduped = new Map();
  normalised.forEach(p => deduped.set(String(p.time), p));
  const sorted = Array.from(deduped.values()).sort((a, b) => {
    const toN = t => typeof t === 'number' ? t : Date.parse(String(t)) / 1000;
    return toN(a.time) - toN(b.time);
  });

  if (!sorted.length) return;

  const candles = sorted.map(p => ({ time: p.time, open: p.open, high: p.high, low: p.low, close: p.close }));
  const volumes = sorted.map((p, i) => ({
    time:  p.time,
    value: p.volume,
    color: i > 0 && p.close < sorted[i-1].close ? 'rgba(239,83,80,0.4)' : 'rgba(38,166,154,0.4)',
  }));

  _candleSeries.setData(candles);
  if (volumes.some(v => v.value > 0)) {
    _volumeSeries.setData(volumes);
  }
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
