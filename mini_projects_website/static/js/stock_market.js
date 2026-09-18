/**
 * stock_market.js
 * ---------------
 * 1. Close-price line chart (fetches /api/stock-data)
 * 2. Model metrics dashboard (fetches /api/model-metrics)
 * 3. Multi-series predictions chart (fetches /api/model-predictions)
 * 4. Live inference simulator powered by saved models (fetches /api/predict)
 * 5. Saved model binary file registry & retraining (fetches /api/model-status)
 */
"use strict";

// ── Color Tokens ─────────────────────────────────────────────────────────────
const C = {
  accent:    "#3b82f6",
  accentLt:  "#60a5fa",
  purple:    "#a78bfa",
  amber:     "#f59e0b",
  white:     "#f0f4ff",
  success:   "#10b981",
  danger:    "#ef4444",
  muted:     "#4d6680",
  surface:   "#0c1524",
  grid:      "rgba(59,130,246,0.08)",
};

// ── Shared Tooltip Config ────────────────────────────────────────────────────
const TOOLTIP = {
  backgroundColor: C.surface,
  borderColor: "rgba(59,130,246,0.35)",
  borderWidth: 1,
  titleColor: "#8ba4c4",
  bodyColor:  C.white,
  padding: 12,
};

function makeGradient(ctx, top, bottom, colorTop, colorBottom) {
  const g = ctx.createLinearGradient(0, top, 0, bottom);
  g.addColorStop(0, colorTop);
  g.addColorStop(1, colorBottom);
  return g;
}

// ══════════════════════════════════════════════════════════════════════════════
//  1. CLOSE-PRICE CHART
// ══════════════════════════════════════════════════════════════════════════════
let stockChart = null;

async function loadCloseChart(n) {
  const status = document.getElementById("chartStatus");
  status.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Loading…';
  try {
    const res  = await fetch(`/api/stock-data?n=${n}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    status.innerHTML = `<i class="fa-solid fa-circle-check" style="color:${C.success}"></i> ${data.count.toLocaleString()} observations`;

    if (stockChart) { stockChart.destroy(); stockChart = null; }
    const ctx = document.getElementById("stockChart").getContext("2d");

    stockChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.dates,
        datasets: [{
          label: "Close Price",
          data: data.closes,
          borderColor: C.accent,
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: C.accentLt,
          pointHoverBorderColor: "#fff",
          fill: true,
          backgroundColor: (ctx2) => {
            const { chartArea } = ctx2.chart;
            if (!chartArea) return "transparent";
            return makeGradient(ctx, chartArea.top, chartArea.bottom,
              "rgba(59,130,246,0.28)", "rgba(59,130,246,0.01)");
          },
          tension: 0.3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 500 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { ...TOOLTIP,
            callbacks: {
              title: i => i[0].label,
              label: i => `  Close: $${i.parsed.y.toLocaleString("en-US",{minimumFractionDigits:2})}`,
            },
          },
        },
        scales: {
          x: { grid: { color: C.grid }, ticks: { color: C.muted, maxTicksLimit: 8, maxRotation: 0, font: { family:"'JetBrains Mono',monospace", size:11 } } },
          y: { grid: { color: C.grid }, ticks: { color: C.muted, font: { family:"'JetBrains Mono',monospace", size:11 },
               callback: v => "$" + v.toLocaleString("en-US",{maximumFractionDigits:0}) } },
        },
      },
    });
  } catch(e) {
    status.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="color:${C.danger}"></i> ${e.message}`;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  2. MODEL METRICS DASHBOARD
// ══════════════════════════════════════════════════════════════════════════════
async function loadModelMetrics() {
  const metaBar  = document.getElementById("metaBar");
  const maeCards = document.getElementById("maeCards");
  const tbody    = document.getElementById("modelTbody");

  try {
    const res  = await fetch("/api/model-metrics");
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const { metrics, meta } = data;

    // ── Meta info bar ───────────────────────────────────────────────────
    metaBar.innerHTML = `
      <i class="fa-solid fa-circle-check" style="color:${C.success}"></i>
      Models trained on <strong>${meta.n_obs.toLocaleString()}</strong> historical days
      &nbsp;|&nbsp; Target: <code>${meta.series}</code>
      &nbsp;|&nbsp; <strong>${meta.models_source}</strong>
    `;

    // ── Find best MAE among runnable models ─────────────────────────────
    const withMAE  = metrics.filter(m => m.mae !== null && m.mae !== undefined);
    const bestMAE  = withMAE.length ? Math.min(...withMAE.map(m => m.mae)) : null;

    // ── Score cards ──────────────────────────────────────────────────
    maeCards.innerHTML = "";
    withMAE.forEach(m => {
      const isBest = m.is_best || (bestMAE !== null && m.mae === bestMAE);
      const cls    = isBest ? "best-card" : "good-card";
      let badge    = isBest ? "🏆 Top Model" : "Trained";
      if (m.accuracy_highlight) {
        badge = `⚡ ${m.accuracy_highlight}`;
      }
      
      const primaryVal = m.dir_accuracy && m.dir_accuracy > 70 
        ? `${m.dir_accuracy.toFixed(1)}% <span style="font-size:0.9rem;font-weight:400;color:var(--text-secondary)">Acc</span>`
        : m.mae.toFixed(6);

      const r2Txt = m.r2_score !== null && m.r2_score !== undefined ? `R²: ${m.r2_score.toFixed(4)}` : "";
      const dirTxt = m.dir_accuracy !== null && m.dir_accuracy !== undefined ? `Dir: ${m.dir_accuracy.toFixed(1)}%` : "";

      maeCards.innerHTML += `
        <div class="mae-card ${cls}">
          <div class="mae-card-model">${m.model}</div>
          <div class="mae-card-value">${primaryVal}</div>
          <span class="mae-card-badge">${badge}</span>
          <div class="mae-card-cv" style="display:flex;justify-content:space-between;align-items:center;">
            <span><i class="fa-solid fa-chart-simple"></i> ${r2Txt || dirTxt}</span>
            <span><i class="fa-solid fa-microchip"></i> ${m.type ? m.type.split(' ')[0] : 'ML'}</span>
          </div>
        </div>`;
    });

    // ── Table rows ──────────────────────────────────────────────────────
    tbody.innerHTML = "";
    metrics.forEach((m, idx) => {
      const mae  = m.mae !== null && m.mae !== undefined
                     ? `<span class="mae-val font-mono">${m.mae.toFixed(6)}</span>`
                     : `<span style="color:var(--text-muted);font-style:italic">—</span>`;
      const rmse = m.rmse !== null && m.rmse !== undefined
                     ? `<span class="font-mono" style="color:#93c5fd">${m.rmse.toFixed(6)}</span>`
                     : `<span style="color:var(--text-muted);font-style:italic">—</span>`;
      const r2   = m.r2_score !== null && m.r2_score !== undefined
                     ? `<span class="font-mono" style="color:${m.r2_score >= 0 ? '#10b981' : '#fca5a5'};font-weight:600">${m.r2_score.toFixed(4)}</span>`
                     : `<span style="color:var(--text-muted);font-style:italic">—</span>`;
      const dir  = m.dir_accuracy !== null && m.dir_accuracy !== undefined
                     ? `<span class="font-mono" style="color:var(--accent-light)">${m.dir_accuracy.toFixed(1)}%</span>`
                     : `<span style="color:var(--text-muted);font-style:italic">—</span>`;

      const isBest = bestMAE !== null && m.mae === bestMAE;
      const statusBadge = m.mae !== null
        ? `<span class="badge badge-done"><i class="fa-solid fa-floppy-disk"></i> Saved .${m.file ? m.file.split('.').pop() : 'bin'}</span>`
        : `<span class="badge badge-soon"><i class="fa-solid fa-triangle-exclamation"></i> External data</span>`;
      const note = m.file ? `<br><small style="color:var(--accent-light);font-family:var(--font-mono)">saved_models/${m.file}</small>` : "";

      tbody.innerHTML += `
        <tr class="${isBest ? "best-row" : ""}">
          <td>${idx + 1}</td>
          <td><span class="model-name">${m.model}${isBest ? " 🏆" : ""}</span>${note}</td>
          <td>${m.desc}</td>
          <td>${mae}</td>
          <td>${rmse}</td>
          <td>${r2}</td>
          <td>${dir}</td>
          <td>${statusBadge}</td>
        </tr>`;
    });

  } catch(e) {
    metaBar.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="color:${C.danger}"></i> Error: ${e.message}`;
    tbody.innerHTML   = `<tr><td colspan="8" class="loading-row" style="color:${C.danger}">${e.message}</td></tr>`;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  3. PREDICTIONS CHART
// ══════════════════════════════════════════════════════════════════════════════
let predChart = null;

const SERIES_CFG = {
  actual:   { label: "Actual ld_Close",   color: C.white,   dash: [] },
  lstm:     { label: "LSTM (Deep Learning)", color: "#10b981", dash: [] },
  rnn:      { label: "Simple RNN",        color: "#06b6d4", dash: [4,2] },
  rf:       { label: "Random Forest",     color: C.accent,  dash: [] },
  arma:     { label: "ARMA(2,7)",         color: C.purple,  dash: [6,3] },
  lr:       { label: "Linear Regression", color: "#38bdf8", dash: [2,2] },
  baseline: { label: "Baseline",          color: C.amber,   dash: [4,4] },
};

function buildDatasets(preds) {
  return Object.entries(SERIES_CFG).map(([key, cfg]) => {
    const vals = preds[key] || [];
    return {
      label: cfg.label,
      data:  vals,
      borderColor: cfg.color,
      borderWidth: key === "actual" ? 2 : 1.5,
      borderDash: cfg.dash,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: false,
      tension: 0.2,
      hidden: false,
      _key: key,
    };
  });
}

async function loadPredChart() {
  const status = document.getElementById("predStatus");
  status.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Loading predictions from saved models…';
  try {
    const res   = await fetch("/api/model-predictions");
    const preds = await res.json();
    if (preds.error) throw new Error(preds.error);

    const n = preds.actual ? preds.actual.length : 0;
    status.innerHTML = `<i class="fa-solid fa-circle-check" style="color:${C.success}"></i> ${n.toLocaleString()} test observations plotted`;

    if (predChart) { predChart.destroy(); predChart = null; }
    const ctx = document.getElementById("predChart").getContext("2d");

    predChart = new Chart(ctx, {
      type: "line",
      data: { labels: preds.dates, datasets: buildDatasets(preds) },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 600 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              color: "#8ba4c4",
              boxWidth: 20,
              font: { size: 12, family:"'Inter',sans-serif" },
            },
          },
          tooltip: { ...TOOLTIP,
            callbacks: {
              title: i => i[0].label,
              label: i => `  ${i.dataset.label}: ${i.parsed.y.toFixed(6)}`,
            },
          },
        },
        scales: {
          x: { grid: { color: C.grid }, ticks: { color: C.muted, maxTicksLimit: 10, maxRotation: 20, font:{family:"'JetBrains Mono',monospace",size:10} } },
          y: { grid: { color: C.grid }, ticks: { color: C.muted, font:{family:"'JetBrains Mono',monospace",size:11}, callback: v => v.toFixed(4) } },
        },
      },
    });

    // ── Toggle buttons ────────────────────────────────────────────────
    document.querySelectorAll(".toggle-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.series;
        const ds  = predChart.data.datasets.find(d => d._key === key);
        if (!ds) return;
        ds.hidden = !ds.hidden;
        btn.classList.toggle("active", !ds.hidden);
        predChart.update();
      });
    });

  } catch(e) {
    status.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="color:${C.danger}"></i> ${e.message}`;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  4. LIVE INFERENCE SIMULATOR (SAVED MODEL PREDICTION)
// ══════════════════════════════════════════════════════════════════════════════
let latestMarketFeatures = null;

async function runLivePrediction() {
  const modelType = document.getElementById("simModelSelect").value;
  const btn = document.getElementById("btnRunPrediction");
  const origBtnText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Running inference…';
  btn.disabled = true;

  const t0 = performance.now();

  try {
    const lag1 = parseFloat(document.getElementById("inputLag1").value) || 0;
    const lag2 = parseFloat(document.getElementById("inputLag2").value) || 0;
    const lag3 = parseFloat(document.getElementById("inputLag3").value) || 0;
    const lag4 = parseFloat(document.getElementById("inputLag4").value) || 0;
    const lag5 = parseFloat(document.getElementById("inputLag5").value) || 0;

    const queryParams = new URLSearchParams({
      model: modelType,
      lag_1: lag1,
      lag_2: lag2,
      lag_3: lag3,
      lag_4: lag4,
      lag_5: lag5,
    });

    const res = await fetch(`/api/predict?${queryParams.toString()}`);
    const data = await res.json();
    const t1 = performance.now();
    const latency = Math.round(t1 - t0);

    if (!data.success) throw new Error(data.error || "Inference failed");

    // Update UI elements
    document.getElementById("simModelName").textContent = data.model_name;
    document.getElementById("simBinaryFile").textContent = data.model_file;
    document.getElementById("simPriceVal").textContent = `$${data.predicted_close_price.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    const sign = data.price_change_dollars >= 0 ? "+" : "";
    document.getElementById("simPriceDelta").textContent = `${sign}$${data.price_change_dollars.toFixed(2)} (${sign}${data.percent_change.toFixed(3)}%)`;
    document.getElementById("simPriceDelta").style.color = data.dir_color;

    document.getElementById("simReturnVal").textContent = `${sign}${data.predicted_log_return.toFixed(6)}`;
    document.getElementById("simPriorClose").textContent = `$${data.last_known_close.toLocaleString("en-US", {minimumFractionDigits: 2})}`;
    document.getElementById("simLatency").textContent = `${latency} ms`;

    const r2Elem = document.getElementById("simR2Score");
    if (r2Elem) {
      r2Elem.textContent = data.r2_score !== null && data.r2_score !== undefined ? data.r2_score.toFixed(4) : "—";
      r2Elem.style.color = (data.r2_score >= 0) ? "#10b981" : "#fca5a5";
    }

    const maeElem = document.getElementById("simTestMAE");
    if (maeElem) {
      maeElem.textContent = data.trained_mae !== null && data.trained_mae !== undefined ? data.trained_mae.toFixed(6) : "—";
    }

    // Direction badge
    const dirBadge = document.getElementById("simDirectionBadge");
    const dirIcon  = document.getElementById("simDirIcon");
    const dirText  = document.getElementById("simDirText");

    dirBadge.style.background = data.predicted_log_return >= 0 ? "rgba(16,185,129,0.15)" : "rgba(239,68,68,0.15)";
    dirBadge.style.borderColor = data.predicted_log_return >= 0 ? "rgba(16,185,129,0.35)" : "rgba(239,68,68,0.35)";
    dirBadge.style.color = data.dir_color;
    dirIcon.className = `fa-solid fa-${data.dir_icon}`;
    dirText.textContent = data.direction;

  } catch(err) {
    alert("Inference Error: " + err.message);
  } finally {
    btn.innerHTML = origBtnText;
    btn.disabled = false;
  }
}

async function populateInitialSimulatorData() {
  try {
    const res = await fetch("/api/predict?model=rf");
    const data = await res.json();
    if (data.success && data.input_features) {
      latestMarketFeatures = data.input_features;
      // API returns return_lag_1 … return_lag_5 (not lag_1 … lag_5)
      const f = data.input_features;
      document.getElementById("inputLag1").value = ((f.return_lag_1)  ?? 0).toFixed(6);
      document.getElementById("inputLag2").value = ((f.return_lag_2)  ?? 0).toFixed(6);
      document.getElementById("inputLag3").value = ((f.return_lag_3)  ?? 0).toFixed(6);
      document.getElementById("inputLag4").value = ((f.return_lag_5)  ?? 0).toFixed(6);
      document.getElementById("inputLag5").value = ((f.return_lag_10) ?? 0).toFixed(6);

      // Auto-run the selected model once data is ready
      runLivePrediction();
    }
  } catch(e) {
    console.warn("Simulator init:", e);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  5. SAVED MODEL STATUS & RETRAINING
// ══════════════════════════════════════════════════════════════════════════════
async function loadSavedModelsStatus() {
  const tbody = document.getElementById("savedModelsTbody");
  if (!tbody) return;

  try {
    const res = await fetch("/api/model-status");
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    tbody.innerHTML = "";
    data.files.forEach(f => {
      const ext = f.name.split('.').pop();
      let badgeCls = "file-badge-joblib";
      if (ext === "pkl") badgeCls = "file-badge-pkl";
      if (ext === "json") badgeCls = "file-badge-json";
      if (ext === "keras") badgeCls = "file-badge-keras";

      tbody.innerHTML += `
        <tr>
          <td><i class="fa-solid fa-file-code" style="color:var(--accent-light);margin-right:8px;"></i><strong class="font-mono">${f.name}</strong></td>
          <td><span class="${badgeCls}">.${ext.toUpperCase()}</span></td>
          <td class="font-mono">${f.size_kb.toLocaleString()} KB</td>
          <td style="color:var(--text-secondary);font-size:0.85rem">${f.modified}</td>
          <td><span class="badge badge-done"><i class="fa-solid fa-circle-check"></i> ${f.status}</span></td>
        </tr>
      `;
    });
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="5" class="loading-row" style="color:${C.danger}">${e.message}</td></tr>`;
  }
}

async function handleRetrain() {
  const btn = document.getElementById("btnRetrain");
  if (!confirm("Retrain and overwrite all saved models in saved_models/?")) return;

  const origText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Retraining models on disk…';
  btn.disabled = true;

  try {
    const res = await fetch("/api/retrain", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      alert("✓ " + data.message);
      loadModelMetrics();
      loadPredChart();
      loadSavedModelsStatus();
      runLivePrediction();
    } else {
      throw new Error(data.error || "Retraining failed");
    }
  } catch(e) {
    alert("Retrain Error: " + e.message);
  } finally {
    btn.innerHTML = origText;
    btn.disabled = false;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  // 1. Close-price chart
  const sel = document.getElementById("rangeSelect");
  if (sel) {
    loadCloseChart(parseInt(sel.value, 10));
    sel.addEventListener("change", () => loadCloseChart(parseInt(sel.value, 10)));
  }

  // 2. Model metrics
  loadModelMetrics();

  // 3. Predictions chart
  loadPredChart();

  // 4. Simulator
  populateInitialSimulatorData();
  const btnRun = document.getElementById("btnRunPrediction");
  if (btnRun) btnRun.addEventListener("click", runLivePrediction);
  
  // Wire model-select AFTER populateInitialSimulatorData so it doesn't fire too early
  const simSel = document.getElementById("simModelSelect");
  if (simSel) simSel.addEventListener("change", runLivePrediction);

  const btnReset = document.getElementById("btnResetLags");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      if (latestMarketFeatures) {
        const f = latestMarketFeatures;
        // Use the correct API key names (return_lag_1 … return_lag_10)
        document.getElementById("inputLag1").value = ((f.return_lag_1)  ?? 0).toFixed(6);
        document.getElementById("inputLag2").value = ((f.return_lag_2)  ?? 0).toFixed(6);
        document.getElementById("inputLag3").value = ((f.return_lag_3)  ?? 0).toFixed(6);
        document.getElementById("inputLag4").value = ((f.return_lag_5)  ?? 0).toFixed(6);
        document.getElementById("inputLag5").value = ((f.return_lag_10) ?? 0).toFixed(6);
        runLivePrediction();
      }
    });
  }

  // 5. Saved models file list & Retrain button
  loadSavedModelsStatus();
  const btnRetrain = document.getElementById("btnRetrain");
  if (btnRetrain) btnRetrain.addEventListener("click", handleRetrain);
});
