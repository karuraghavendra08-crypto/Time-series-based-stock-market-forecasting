/**
 * TIME SERIES BASED STOCK MARKET FORECASTING USING PYTHON
 * Frontend Controller — Quantitative Research Laboratory & Forecaster
 * Features:
 *   - Genuine Yahoo Finance streaming & historical OHLCV chart rendering
 *   - Safe Chart.js lifecycle management with zero canvas memory leaks
 *   - Monotonic request-sequence tracking preventing out-of-order race conditions
 *   - Dynamic telemetry badge reflecting live, cached, or unavailable status
 *   - Strict separation of Live Yahoo streaming from historical CSV research
 *   - Honest prediction breakdown distinguishing classification from regression
 */

// ── 1. Unified Application State ─────────────────────────────────────────────
const appState = {
  ticker: "^DJI",
  source: "Yahoo Finance",
  historicalDates: [],
  historicalPrices: [],
  unit: "pts",
  baselinePrice: null,
  baselineTimestamp: null,
  marketDataTimestamp: null,
  isStale: false,
  currentModel: "bull_bear_regime",
  currentDatasetKey: "modern",
  marketRequestId: 0,
  inferenceRequestId: 0,
  forecastPoint: null, // { price, direction, color }
};

let mainChart = null;
let labAccuracyChart = null;
let labAucChart = null;
let labMetricsCache = [];
let pollIntervalTimer = null;

// ── 2. Initialization ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initializeDashboard();
});

async function initializeDashboard() {
  try {
    // 1. Load initial dataset metadata for verification strip
    await switchDatasetMetadata("modern", false);

    // 2. Fetch live Yahoo market data & render main chart
    await fetchAndRenderMarketData(true);

    // 3. Run initial inference for prediction breakdown
    await runInference();

    // 4. Load Model Laboratory metrics table and charts
    await loadModelLaboratory();

    // 5. Setup periodic background refresh (60s interval, visible tabs only)
    setupBackgroundPolling();
  } catch (err) {
    console.error("Dashboard initialization notice:", err);
  }
}

// ── 3. Dynamic Dataset Metadata (Auditing Strip) ──────────────────────────────
async function switchDatasetMetadata(datasetKey, reloadChart = false) {
  appState.currentDatasetKey = datasetKey;

  const btnModern = document.getElementById("btnModernDataset");
  const btnHist = document.getElementById("btnHistoricalDataset");
  if (btnModern && btnHist) {
    btnModern.classList.toggle("active", datasetKey === "modern");
    btnHist.classList.toggle("active", datasetKey === "historical");
  }

  try {
    const res = await fetch(`/api/dataset-metadata?dataset=${datasetKey}`);
    const data = await res.json();
    if (data.status === "Unavailable" || data.status === "Error") {
      throw new Error(data.message || "Dataset unavailable");
    }

    const titleElem = document.getElementById("metaDatasetTitle");
    const obsElem = document.getElementById("metaValidObs");
    const dateRangeElem = document.getElementById("metaDateRange");
    const qualityElem = document.getElementById("metaQuality");
    const closeElem = document.getElementById("metaLastClose");
    const highLowElem = document.getElementById("metaHighLow");

    if (titleElem) titleElem.textContent = data.name;
    if (obsElem) obsElem.textContent = Number(data.valid_observations).toLocaleString();
    if (dateRangeElem) dateRangeElem.innerHTML = `${data.first_date_display} &rarr; ${data.last_date_display}`;
    if (qualityElem) qualityElem.textContent = `${data.duplicate_records} Duplicates • ${data.missing_records} NaN`;
    if (closeElem) closeElem.textContent = `${Number(data.latest_close).toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${data.unit}`;
    if (highLowElem) highLowElem.textContent = `${Number(data.highest_close).toLocaleString()} / ${Number(data.lowest_close).toLocaleString()} ${data.unit}`;

    // Note: Changing historical dataset metadata does NOT overwrite the live forecasting chart.
    if (reloadChart) {
      await loadHistoricalCsvChart(datasetKey);
    }
  } catch (err) {
    console.warn("Dataset metadata fetch error:", err);
  }
}

// ── 4. Live Market Data Fetching & Chart Rendering ───────────────────────────
async function fetchAndRenderMarketData(isInitial = false) {
  const pill = document.getElementById("livePricePill");
  const refreshBtn = document.getElementById("btnRefreshQuote");

  if (pill && isInitial) {
    pill.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Fetching quote…';
  }
  if (refreshBtn) refreshBtn.disabled = true;

  const reqId = ++appState.marketRequestId;

  try {
    const res = await fetch(`/api/live-market-data?ticker=${appState.ticker}`);
    const data = await res.json();

    if (reqId !== appState.marketRequestId) return; // Prevent race conditions

    if (res.ok && data.success) {
      appState.source = data.source || "Yahoo Finance";
      appState.historicalDates = data.history_dates || data.recent_dates || [];
      appState.historicalPrices = data.history_closes || data.recent_closes || [];
      appState.baselinePrice = data.last_close;
      appState.unit = data.unit || "$";
      appState.marketDataTimestamp = data.last_data_timestamp;
      appState.isStale = Boolean(data.is_stale);

      updateTelemetryBadge({
        status: appState.isStale ? "stale" : "connected",
        ticker: data.ticker,
        interval: data.data_interval || "1d",
        timestamp: data.last_data_timestamp,
        method: data.source_method,
      });

      updatePricePill(data);
      renderChartWithForecast(appState.forecastPoint);
    } else {
      updateTelemetryBadge({
        status: "error",
        ticker: appState.ticker,
        message: data.error || "Yahoo Finance data unavailable",
      });

      if (pill) {
        pill.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Live data unavailable</span>`;
      }
      const statusElem = document.getElementById("chartObservationsCount");
      if (statusElem) {
        statusElem.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> ${data.error || "Live data unavailable"}</span>`;
      }
    }
  } catch (err) {
    if (reqId !== appState.marketRequestId) return;
    updateTelemetryBadge({
      status: "error",
      ticker: appState.ticker,
      message: err.message,
    });
    if (pill) {
      pill.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Network error</span>`;
    }
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function updateTelemetryBadge(state) {
  const dot = document.getElementById("systemTelemetryDot");
  if (!dot) return;

  if (state.status === "connected") {
    dot.innerHTML = `<span style="color:var(--acid-lime);font-size:11px;">●</span> Connected &bull; ${state.ticker} (${state.interval}) &bull; ${state.timestamp || 'Live'}`;
    dot.title = `Source: Yahoo Finance (${state.method || 'yfinance'}). Last observation: ${state.timestamp}`;
  } else if (state.status === "stale") {
    dot.innerHTML = `<span style="color:#FFAA00;font-size:11px;">●</span> Cached Yahoo Data &bull; ${state.ticker} (${state.timestamp})`;
    dot.title = "Serving cached Yahoo Finance payload. Upstream service delayed.";
  } else {
    dot.innerHTML = `<span style="color:#FF725E;font-size:11px;">●</span> Yahoo Finance Offline &bull; ${state.ticker}`;
    dot.title = state.message || "Failed to fetch market observations from Yahoo Finance.";
  }
}

function updatePricePill(data) {
  const pill = document.getElementById("livePricePill");
  if (!pill) return;
  const sign = data.day_change >= 0 ? "+" : "";
  const colorStyle = data.day_change >= 0 ? "color:#2D6B00;" : "color:#B82512;";
  const unit = data.unit || "$";
  pill.innerHTML = `
    <strong>${data.ticker}</strong>: ${data.last_close.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}
    <span style="${colorStyle}; font-weight:600; margin-left:4px;">(${sign}${data.day_pct_change}%)</span>
  `;
}

// ── 5. Main Forecast Chart Management ─────────────────────────────────────────
function renderChartWithForecast(forecastPoint = null) {
  const canvas = document.getElementById("mainForecastChart");
  if (!canvas) return;

  const dates = [...appState.historicalDates];
  const prices = [...appState.historicalPrices];
  const unit = appState.unit || "pts";

  if (dates.length === 0 || prices.length === 0) {
    return;
  }

  const datasets = [
    {
      label: `${appState.ticker} ${appState.source} (${unit})`,
      data: prices,
      borderColor: "#465BFF", // Electric Blue
      backgroundColor: "rgba(70, 91, 255, 0.05)",
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: true,
      tension: 0.15,
    }
  ];

  const labels = [...dates];

  if (forecastPoint && forecastPoint.price !== null && forecastPoint.price !== undefined) {
    labels.push("Forward (t+1)");
    const forecastSeries = new Array(prices.length).fill(null);
    forecastSeries[forecastSeries.length - 1] = prices[prices.length - 1]; // Anchor
    forecastSeries.push(forecastPoint.price);

    datasets.push({
      label: `Model Forecast (${forecastPoint.direction || 'Target'})`,
      data: forecastSeries,
      borderColor: forecastPoint.color || "#B8E64B",
      backgroundColor: "transparent",
      borderWidth: 2.5,
      borderDash: [5, 4],
      pointRadius: (ctx) => (ctx.dataIndex === labels.length - 1 ? 6 : 0),
      pointBackgroundColor: forecastPoint.color || "#B8E64B",
      pointBorderColor: "#FFFFFF",
      pointBorderWidth: 2,
      tension: 0,
    });
  }

  if (mainChart) {
    mainChart.data.labels = labels;
    mainChart.data.datasets = datasets;
    mainChart.options.scales.y.ticks.callback = (val) => `${val.toLocaleString()} ${unit}`;
    mainChart.update();
  } else {
    const ctx = canvas.getContext("2d");
    mainChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { boxWidth: 12, font: { family: "Inter", size: 12 }, color: "#111216" },
          },
          tooltip: {
            backgroundColor: "#111216",
            titleFont: { family: "JetBrains Mono", size: 12 },
            bodyFont: { family: "Inter", size: 12 },
            padding: 10,
            callbacks: {
              label: (context) => {
                const val = context.parsed.y;
                return val !== null
                  ? ` ${context.dataset.label}: ${val.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`
                  : "";
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(17, 18, 22, 0.04)" },
            ticks: { font: { family: "JetBrains Mono", size: 10 }, color: "#676A73", maxTicksLimit: 8 },
          },
          y: {
            grid: { color: "rgba(17, 18, 22, 0.05)" },
            ticks: {
              font: { family: "JetBrains Mono", size: 11 },
              color: "#676A73",
              callback: (val) => `${val.toLocaleString()} ${unit}`,
            },
          },
        },
      },
    });
  }

  const statusElem = document.getElementById("chartObservationsCount");
  if (statusElem) {
    const firstDate = dates[0] || "";
    const lastDate = dates[dates.length - 1] || "";
    const staleNote = appState.isStale ? " (Cached)" : "";
    statusElem.textContent = `${prices.length} observations plotted (${firstDate} → ${lastDate}) • ${appState.source}${staleNote}`;
  }
}

// Optional: explicit historical CSV inspection
async function loadHistoricalCsvChart(datasetKey) {
  const statusElem = document.getElementById("chartObservationsCount");
  if (statusElem) statusElem.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Loading CSV series…';

  try {
    const res = await fetch(`/api/stock-data?dataset=${datasetKey}&n=150`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    appState.source = `Historical CSV (${datasetKey})`;
    appState.historicalDates = data.dates || [];
    appState.historicalPrices = data.closes || [];
    appState.unit = data.unit || "pts";
    appState.forecastPoint = null;

    renderChartWithForecast(null);
  } catch (err) {
    if (statusElem) {
      statusElem.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Error loading CSV: ${err.message}</span>`;
    }
  }
}

// ── 6. Market Quote & Ticker Selection ────────────────────────────────────────
async function onInstrumentChange() {
  const selectElem = document.getElementById("selectTicker");
  if (!selectElem) return;
  appState.ticker = selectElem.value;
  appState.forecastPoint = null; // Clear forward overlay
  await fetchAndRenderMarketData(true);
  await runInference();
}

async function refreshMarketQuote() {
  await fetchAndRenderMarketData(false);
}

function setupBackgroundPolling() {
  if (pollIntervalTimer) clearInterval(pollIntervalTimer);
  pollIntervalTimer = setInterval(() => {
    if (document.visibilityState === "visible") {
      fetchAndRenderMarketData(false);
    }
  }, 60000);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      fetchAndRenderMarketData(false);
    }
  });
}

// ── 7. Live Prediction & Breakdown Panel ──────────────────────────────────────
function onModelChange() {
  const select = document.getElementById("selectModel");
  if (!select) return;
  appState.currentModel = select.value;
  runInference();
}

async function runInference() {
  const btn = document.getElementById("btnRunInference");
  if (btn) {
    btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Computing…';
    btn.disabled = true;
  }

  const reqId = ++appState.inferenceRequestId;

  try {
    const query = new URLSearchParams({
      model: appState.currentModel,
      ticker: appState.ticker,
      use_live_market: "true",
    });

    const res = await fetch(`/api/predict?${query.toString()}`);
    const data = await res.json();

    if (reqId !== appState.inferenceRequestId) return; // Discard outdated response

    if (res.status === 400 && data.unsupported_instrument) {
      handleUnsupportedInstrument(data);
      return;
    }

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Inference failed");
    }

    // 1. Update Regime Banner
    const banner = document.getElementById("predRegimeBanner");
    const taskLabel = document.getElementById("predTaskTypeLabel");
    const dirSymbol = document.getElementById("predDirSymbol");
    const dirText = document.getElementById("predDirectionText");
    const probVal = document.getElementById("predProbValue");
    const probLabel = document.getElementById("predProbLabel");

    const isBull = data.prob_up >= 0.5;
    if (banner) {
      banner.className = `regime-banner ${isBull ? "bullish" : "bearish"}`;
    }
    if (taskLabel) taskLabel.textContent = `${data.target_name} (${data.forecast_horizon})`;
    if (dirSymbol) dirSymbol.textContent = data.dir_symbol || (isBull ? "▲" : "▼");
    if (dirText) dirText.textContent = data.direction;
    if (probVal) probVal.textContent = `${data.confidence_pct}%`;
    if (probLabel) probLabel.textContent = isBull ? "Bullish Probability" : "Bearish Probability";

    // 2. Numerical Forecast Grid (Continuous Models vs Classification Models)
    const baseElem = document.getElementById("predBaselineClose");
    const targetElem = document.getElementById("predTargetClose");
    const deltaElem = document.getElementById("predPriceDelta");
    const pctElem = document.getElementById("predPercentDelta");

    const unit = data.unit || appState.unit;
    if (baseElem) {
      baseElem.textContent = `${data.baseline_close.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;
    }

    if (data.is_numerical && data.predicted_target_price !== null) {
      if (targetElem) {
        targetElem.textContent = `${data.predicted_target_price.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;
      }
      const deltaSign = data.price_change >= 0 ? "+" : "";
      if (deltaElem) {
        deltaElem.textContent = `${deltaSign}${data.price_change.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;
        deltaElem.className = `num-cell-value font-mono ${data.price_change >= 0 ? "delta-pos" : "delta-neg"}`;
      }
      if (pctElem) {
        pctElem.textContent = `${deltaSign}${data.percent_change.toFixed(3)}%`;
        pctElem.className = `num-cell-value font-mono ${data.percent_change >= 0 ? "delta-pos" : "delta-neg"}`;
      }

      // Overlay forecast point onto existing live chart
      appState.forecastPoint = {
        price: data.predicted_target_price,
        direction: data.direction,
        color: isBull ? "#B8E64B" : "#FF725E",
      };
      renderChartWithForecast(appState.forecastPoint);
    } else {
      // Classification models do not predict continuous target prices
      if (targetElem) {
        targetElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">Directional State Only</span>`;
      }
      if (deltaElem) {
        deltaElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">N/A (Classification)</span>`;
        deltaElem.className = "num-cell-value font-mono";
      }
      if (pctElem) {
        pctElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">N/A (Probability Only)</span>`;
        pctElem.className = "num-cell-value font-mono";
      }

      // Remove any previously projected numerical forward step
      appState.forecastPoint = null;
      renderChartWithForecast(null);
    }

    // 3. Plain-English Interpretation
    const interpElem = document.getElementById("predInterpretationText");
    if (interpElem) interpElem.textContent = data.interpretation;

    // 4. Feature Drivers List
    const featList = document.getElementById("featuresDriversList");
    if (featList) {
      if (data.feature_importance && data.feature_importance.length > 0) {
        featList.innerHTML = "";
        const maxImp = Math.max(...data.feature_importance.map((f) => f.importance || 0.1), 0.01);
        data.feature_importance.forEach((f) => {
          const barPct = Math.round((f.importance / maxImp) * 100);
          featList.innerHTML += `
            <div class="feature-bar-row">
              <span class="feature-name">${f.feature}</span>
              <div class="feature-bar-track"><div class="feature-bar-fill" style="width: ${barPct}%;"></div></div>
              <span class="feature-weight font-mono">${(f.importance * 100).toFixed(1)}%</span>
            </div>`;
        });
      } else {
        const taskNote = data.task_type === "return_regression"
          ? "ARMA is an econometrics time-series model; Gini importance is not applicable. Predictions use auto-regressive and moving-average terms of historical log-returns."
          : data.task_type === "benchmark"
          ? "Naive benchmark always predicts majority class; no feature importance applies."
          : data.task_type && data.task_type.includes("deep_learning")
          ? "Deep learning sequence models do not expose tree Gini importance. Predictions rely on the 30-day normalized OHLCV feature sequence."
          : "Feature importance not available for this model.";
        featList.innerHTML = `<div style="color: var(--text-secondary); font-size: 12.5px; padding: 8px 0;">${taskNote}</div>`;
      }
    }

    // 5. Test Context
    const testSampleElem = document.getElementById("predTestSampleCount");
    const testAccElem = document.getElementById("predTestAcc");
    if (testSampleElem) testSampleElem.textContent = `${data.test_sample_count} test samples`;
    if (testAccElem) testAccElem.textContent = data.test_accuracy ? `${data.test_accuracy.toFixed(2)}%` : "N/A (Continuous)";

  } catch (err) {
    const interpElem = document.getElementById("predInterpretationText");
    if (interpElem) {
      interpElem.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Inference Error: ${err.message}</span>`;
    }
  } finally {
    if (btn) {
      btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Forecast';
      btn.disabled = false;
    }
  }
}

function handleUnsupportedInstrument(data) {
  // Clear numerical forward overlay
  appState.forecastPoint = null;
  renderChartWithForecast(null);

  const banner = document.getElementById("predRegimeBanner");
  if (banner) {
    banner.className = "regime-banner bearish";
  }
  const taskLabel = document.getElementById("predTaskTypeLabel");
  if (taskLabel) taskLabel.textContent = "Model Scope & Research Boundary";
  const dirSymbol = document.getElementById("predDirSymbol");
  if (dirSymbol) dirSymbol.textContent = "ℹ";
  const dirText = document.getElementById("predDirectionText");
  if (dirText) dirText.textContent = "UNVALIDATED INSTRUMENT";
  const probVal = document.getElementById("predProbValue");
  if (probVal) probVal.textContent = "N/A";
  const probLabel = document.getElementById("predProbLabel");
  if (probLabel) probLabel.textContent = "Dow Jones Validated Only";

  const baseElem = document.getElementById("predBaselineClose");
  const targetElem = document.getElementById("predTargetClose");
  const deltaElem = document.getElementById("predPriceDelta");
  const pctElem = document.getElementById("predPercentDelta");

  if (baseElem && appState.baselinePrice) {
    baseElem.textContent = `${appState.baselinePrice.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${appState.unit}`;
  }
  if (targetElem) targetElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">Unvalidated Instrument</span>`;
  if (deltaElem) {
    deltaElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">N/A</span>`;
    deltaElem.className = "num-cell-value font-mono";
  }
  if (pctElem) {
    pctElem.innerHTML = `<span style="color:var(--text-secondary);font-size:12px;">N/A</span>`;
    pctElem.className = "num-cell-value font-mono";
  }

  const interpElem = document.getElementById("predInterpretationText");
  if (interpElem) {
    interpElem.innerHTML = `
      <span style="color:#FFAA00;"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Research Boundary Notice:</strong></span>
      The saved machine learning and econometric models in this repository were trained and evaluated exclusively on the
      historical Dow Jones Industrial Average (^DJI). While genuine Yahoo Finance market data and price charts are fully supported
      for <strong>${data.ticker}</strong>, generating model forecasts for unvalidated instruments would violate quantitative research integrity.
      Please select <strong>Dow Jones (^DJI)</strong> from the instrument dropdown to execute audited forward forecasts.
    `;
  }

  const featList = document.getElementById("featuresDriversList");
  if (featList) {
    featList.innerHTML = `<div style="color: var(--text-secondary); font-size: 12.5px; padding: 8px 0;">Model inference disabled for ${data.ticker}. Select Dow Jones (^DJI) to inspect feature drivers.</div>`;
  }

  const testSampleElem = document.getElementById("predTestSampleCount");
  const testAccElem = document.getElementById("predTestAcc");
  if (testSampleElem) testSampleElem.textContent = "N/A (Dow Jones Only)";
  if (testAccElem) testAccElem.textContent = "N/A";
}

// ── 8. Model Laboratory ───────────────────────────────────────────────────────
function switchLabTab(tabKey) {
  const btnTable = document.getElementById("btnTabTable");
  const btnCharts = document.getElementById("btnTabCharts");
  const contentTable = document.getElementById("labTabContentTable");
  const contentCharts = document.getElementById("labTabContentCharts");

  if (btnTable && btnCharts) {
    btnTable.classList.toggle("active", tabKey === "table");
    btnCharts.classList.toggle("active", tabKey === "charts");
  }
  if (contentTable && contentCharts) {
    contentTable.style.display = tabKey === "table" ? "block" : "none";
    contentCharts.style.display = tabKey === "charts" ? "block" : "none";
  }

  if (tabKey === "charts") {
    setTimeout(renderLabCharts, 50);
  }
}

async function loadModelLaboratory() {
  const tbody = document.getElementById("labTableBody");
  try {
    const res = await fetch("/api/model-metrics");
    const data = await res.json();
    if (!data.models) throw new Error("No model metrics returned");

    labMetricsCache = data.models;
    tbody.innerHTML = "";

    data.models.forEach((m) => {
      const isBest = m.is_best === true;
      const accStr = m.dir_accuracy !== null && m.dir_accuracy !== undefined ? `${m.dir_accuracy.toFixed(2)}%` : "—";
      const aucStr = m.roc_auc !== null && m.roc_auc !== undefined ? m.roc_auc.toFixed(4) : "—";
      const precStr = m.precision !== null && m.precision !== undefined ? `${m.precision.toFixed(2)}%` : "—";
      const recStr = m.recall !== null && m.recall !== undefined ? `${m.recall.toFixed(2)}%` : "—";
      const f1Str = m.f1_score !== null && m.f1_score !== undefined ? m.f1_score.toFixed(4) : "—";
      const maeStr = m.mae !== null && m.mae !== undefined ? `${m.mae.toFixed(6)}` : "N/A";

      const taskClass = m.task === "classification" ? "classification" : m.task === "regression" ? "regression" : "benchmark";

      tbody.innerHTML += `
        <tr class="${isBest ? "highlight-best" : ""}">
          <td>
            <strong>${m.model}</strong>${isBest ? ' <span style="color:#2D5200;">🏆 Best</span>' : ''}
            <br><small class="font-mono" style="color:var(--text-muted); font-size:11px;">saved_models/${m.file}</small>
          </td>
          <td>${m.type}</td>
          <td><span class="task-tag ${taskClass}">${m.task || 'Model'}</span></td>
          <td class="font-mono"><strong>${accStr}</strong></td>
          <td class="font-mono">${aucStr}</td>
          <td class="font-mono">${precStr}</td>
          <td class="font-mono">${recStr}</td>
          <td class="font-mono">${f1Str}</td>
          <td class="font-mono" style="color:${m.mae !== null ? '#1E3A8A' : 'var(--text-muted)'};">${maeStr}</td>
          <td><span style="font-size:11px; color:#2D6B00;"><i class="fa-solid fa-circle-check"></i> Out-of-Sample Verified</span></td>
        </tr>`;
    });

  } catch (err) {
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="10" style="color:#FF725E; text-align:center; padding:20px;">Failed to load metrics: ${err.message}</td></tr>`;
    }
  }
}

function renderLabCharts() {
  if (!labMetricsCache || labMetricsCache.length === 0) return;

  const classificationModels = labMetricsCache.filter((m) => m.dir_accuracy !== null);
  const labels = classificationModels.map((m) => m.model.split(" ")[0]);
  const accValues = classificationModels.map((m) => m.dir_accuracy);
  const aucModels = classificationModels.filter((m) => m.roc_auc !== null);
  const aucLabels = aucModels.map((m) => m.model.split(" ")[0]);
  const aucValues = aucModels.map((m) => m.roc_auc);

  // 1. Accuracy Comparison Bar Chart
  const ctxAcc = document.getElementById("labAccuracyChart");
  if (ctxAcc) {
    if (labAccuracyChart) labAccuracyChart.destroy();
    labAccuracyChart = new Chart(ctxAcc.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Direction Accuracy (%)",
            data: accValues,
            backgroundColor: accValues.map((v) => (v >= 90 ? "#B8E64B" : v >= 75 ? "#465BFF" : "#DFE1E6")),
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Accuracy: ${ctx.parsed.y.toFixed(2)}%`,
            },
          },
        },
        scales: {
          y: {
            min: 50,
            max: 100,
            ticks: { font: { family: "JetBrains Mono", size: 10 }, callback: (v) => `${v}%` },
            grid: { color: "rgba(17,18,22,0.05)" },
          },
          x: {
            ticks: { font: { family: "Inter", size: 10 } },
            grid: { display: false },
          },
        },
      },
    });
  }

  // 2. ROC-AUC Bar Chart
  const ctxAuc = document.getElementById("labAucChart");
  if (ctxAuc) {
    if (labAucChart) labAucChart.destroy();
    labAucChart = new Chart(ctxAuc.getContext("2d"), {
      type: "bar",
      data: {
        labels: aucLabels,
        datasets: [
          {
            label: "ROC-AUC Score",
            data: aucValues,
            backgroundColor: "#2D6B00",
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ROC-AUC: ${ctx.parsed.y.toFixed(4)}`,
            },
          },
        },
        scales: {
          y: {
            min: 0.4,
            max: 1.0,
            ticks: { font: { family: "JetBrains Mono", size: 10 } },
            grid: { color: "rgba(17,18,22,0.05)" },
          },
          x: {
            ticks: { font: { family: "Inter", size: 10 } },
            grid: { display: false },
          },
        },
      },
    });
  }
}
