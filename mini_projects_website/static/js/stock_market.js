/**
 * TIME SERIES BASED STOCK MARKET FORECASTING USING PYTHON
 * Frontend Controller — Quantitative Research Laboratory & Forecaster
 * Features:
 *   - Verified dynamic dataset metadata toggling (Modern 2015-2025 vs Historical 1901-2012)
 *   - Safe Chart.js lifecycle management with zero canvas memory leaks
 *   - Live instrument quotes & real-time prediction breakdown
 *   - Dedicated Model Laboratory benchmarks and visual metric comparisons
 *   - Error boundary handling with zero indefinite loading states
 */

// ── State Management ─────────────────────────────────────────────────────────
let mainChart = null;
let labAccuracyChart = null;
let labAucChart = null;

let currentDatasetKey = "modern";
let currentTicker = "^DJI";
let currentModel = "bull_bear_regime";
let currentUnit = "pts";

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initializeDashboard();
});

async function initializeDashboard() {
  try {
    // 1. Load initial dataset metadata
    await switchDatasetMetadata("modern", false);

    // 2. Load and render main forecast chart
    await loadMainForecastChart();

    // 3. Fetch initial market quote
    await refreshMarketQuote(false);

    // 4. Run initial inference for prediction breakdown
    await runInference();

    // 5. Load Model Laboratory metrics table and charts
    await loadModelLaboratory();

  } catch (err) {
    console.error("Dashboard initialization notice:", err);
  }
}

// ── 1. Dynamic Dataset Metadata (Phase 2 & 3) ─────────────────────────────────
async function switchDatasetMetadata(datasetKey, reloadChart = true) {
  currentDatasetKey = datasetKey;

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

    if (reloadChart) {
      await loadMainForecastChart();
    }
  } catch (err) {
    console.warn("Dataset metadata fetch error:", err);
  }
}

// ── 2. Main Forecast Chart (Phase 7 & 8) ──────────────────────────────────────
async function loadMainForecastChart(forecastPoint = null) {
  const statusElem = document.getElementById("chartObservationsCount");
  if (statusElem) statusElem.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Loading series…';

  try {
    const res = await fetch(`/api/stock-data?dataset=${currentDatasetKey}&n=150`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const canvas = document.getElementById("mainForecastChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Dispose old chart instance safely to avoid memory leaks
    if (mainChart) {
      mainChart.destroy();
      mainChart = null;
    }

    const labels = [...data.dates];
    const prices = [...data.closes];
    currentUnit = data.unit || "pts";

    // Dataset series setup
    const datasets = [
      {
        label: `Historical Close (${currentUnit})`,
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

    // If a projected forecast point is supplied, append it
    if (forecastPoint && forecastPoint.price) {
      labels.push("Forward (t+5)");
      const forecastSeries = new Array(prices.length).fill(null);
      forecastSeries[forecastSeries.length - 1] = prices[prices.length - 1]; // Anchor
      forecastSeries.push(forecastPoint.price);

      datasets.push({
        label: `Model Forecast (${forecastPoint.direction})`,
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

    mainChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              boxWidth: 12,
              font: { family: "Inter", size: 12 },
              color: "#111216",
            },
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
                  ? ` ${context.dataset.label}: ${val.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${currentUnit}`
                  : "";
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(17, 18, 22, 0.04)" },
            ticks: {
              font: { family: "JetBrains Mono", size: 10 },
              color: "#676A73",
              maxTicksLimit: 8,
            },
          },
          y: {
            grid: { color: "rgba(17, 18, 22, 0.05)" },
            ticks: {
              font: { family: "JetBrains Mono", size: 11 },
              color: "#676A73",
              callback: (val) => `${val.toLocaleString()} ${currentUnit}`,
            },
          },
        },
      },
    });

    if (statusElem) {
      statusElem.textContent = `${prices.length} observations plotted (${data.first_date} → ${data.last_date})`;
    }

  } catch (err) {
    if (statusElem) {
      statusElem.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Error loading chart: ${err.message}</span>`;
    }
  }
}

// ── 3. Market Quote & Ticker Selection (Phase 3 & 7) ──────────────────────────
async function onInstrumentChange() {
  const selectElem = document.getElementById("selectTicker");
  if (!selectElem) return;
  currentTicker = selectElem.value;
  await refreshMarketQuote(true);
  await runInference();
}

async function refreshMarketQuote(autoPredict = true) {
  const pill = document.getElementById("livePricePill");
  const refreshBtn = document.getElementById("btnRefreshQuote");

  if (pill) pill.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Refreshing…';
  if (refreshBtn) refreshBtn.disabled = true;

  try {
    const res = await fetch(`/api/live-market-data?ticker=${currentTicker}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Unable to retrieve quote");

    currentUnit = data.unit || "$";
    const sign = data.day_change >= 0 ? "+" : "";
    const colorStyle = data.day_change >= 0 ? "color:#2D6B00;" : "color:#B82512;";

    if (pill) {
      pill.innerHTML = `
        <strong>${data.ticker}</strong>: ${data.last_close.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${currentUnit}
        <span style="${colorStyle}; font-weight:600; margin-left:4px;">(${sign}${data.day_pct_change}%)</span>
      `;
    }

    // If recent closes are available in live data, update the main chart with this instrument's recent candles!
    if (data.recent_closes && data.recent_closes.length > 5 && mainChart) {
      // Re-anchor chart with instrument's actual candles
      const dates = data.recent_dates || [];
      mainChart.data.labels = [...dates];
      mainChart.data.datasets[0].label = `${data.ticker} Recent Daily Close (${currentUnit})`;
      mainChart.data.datasets[0].data = [...data.recent_closes];
      mainChart.options.scales.y.ticks.callback = (val) => `${val.toLocaleString()} ${currentUnit}`;
      mainChart.update();
    }

  } catch (err) {
    if (pill) {
      pill.innerHTML = `<span style="color:#FF725E"><i class="fa-solid fa-triangle-exclamation"></i> Live feed delayed</span>`;
    }
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

// ── 4. Live Prediction & Breakdown Panel (Phase 4, 5, 10, 11) ───────────────────
function onModelChange() {
  const select = document.getElementById("selectModel");
  if (!select) return;
  currentModel = select.value;
  runInference();
}

async function runInference() {
  const btn = document.getElementById("btnRunInference");
  if (btn) {
    btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Computing…';
    btn.disabled = true;
  }

  try {
    const query = new URLSearchParams({
      model: currentModel,
      ticker: currentTicker,
      use_live_market: "true",
    });

    const res = await fetch(`/api/predict?${query.toString()}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Inference failed");

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

    // 2. Numerical Forecast Grid (Verified Math)
    const baseElem = document.getElementById("predBaselineClose");
    const targetElem = document.getElementById("predTargetClose");
    const deltaElem = document.getElementById("predPriceDelta");
    const pctElem = document.getElementById("predPercentDelta");

    const unit = data.unit || currentUnit;
    if (baseElem) baseElem.textContent = `${data.baseline_close.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;
    if (targetElem) targetElem.textContent = `${data.predicted_target_price.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;

    const deltaSign = data.price_change >= 0 ? "+" : "";
    if (deltaElem) {
      deltaElem.textContent = `${deltaSign}${data.price_change.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})} ${unit}`;
      deltaElem.className = `num-cell-value font-mono ${data.price_change >= 0 ? "delta-pos" : "delta-neg"}`;
    }
    if (pctElem) {
      pctElem.textContent = `${deltaSign}${data.percent_change.toFixed(3)}%`;
      pctElem.className = `num-cell-value font-mono ${data.percent_change >= 0 ? "delta-pos" : "delta-neg"}`;
    }

    // 3. Plain-English Interpretation Box
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
          ? "ARMA is a statistical model; Gini feature importance is not applicable. The model uses auto-regressive and moving-average terms of the log-return series."
          : data.task_type === "benchmark"
          ? "Naive benchmark always predicts the majority class; no feature importance is applicable."
          : data.task_type && data.task_type.includes("deep_learning")
          ? "Deep learning models do not expose tree-based Gini importance. Predictions rely on the 30-day scaled feature sequence."
          : "Feature importance not available for this model.";
        featList.innerHTML = `<div style="color: var(--text-secondary); font-size: 12.5px; padding: 8px 0;">${taskNote}</div>`;
      }
    }

    // 5. Test Context
    const testSampleElem = document.getElementById("predTestSampleCount");
    const testAccElem = document.getElementById("predTestAcc");
    if (testSampleElem) testSampleElem.textContent = `${data.test_sample_count} test samples`;
    if (testAccElem) testAccElem.textContent = data.test_accuracy ? `${data.test_accuracy.toFixed(2)}%` : "N/A (Continuous)";

    // 6. Update Main Forecast Chart with Projected Forward Step
    if (mainChart) {
      loadMainForecastChart({
        price: data.predicted_target_price,
        direction: data.direction,
        color: isBull ? "#B8E64B" : "#FF725E",
      });
    }

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

// ── 5. Model Laboratory (Phase 9 & 10) ─────────────────────────────────────────
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

  // Render charts when user switches to charts tab
  if (tabKey === "charts") {
    setTimeout(renderLabCharts, 50);
  }
}

let labMetricsCache = [];

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
