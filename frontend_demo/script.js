const settingsBtn = document.getElementById("settingsBtn");
const settingsPopup = document.getElementById("settingsPopup");
const updateSlider = document.getElementById("updateSlider");
const percentDisplay = document.getElementById("percentDisplay");
const stockBox = document.getElementById("stock-box");
const priceBox = document.getElementById("price-box");
const refreshBtn = document.getElementById("refreshBtn");

const API_BASE = "http://127.0.0.1:8001";

async function fetchPredictions() {
  // Expect API contract: { predictions: [{symbol, stock_name, confidence, horizon, model_version}], count }
  const res = await fetch(`${API_BASE}/predictions?limit=10`);
  if (!res.ok) throw new Error(`predictions failed: ${res.status}`);
  return await res.json();
}

async function fetchLatestClose(symbol) {
  const res = await fetch(`${API_BASE}/stocks/${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(`stocks/${symbol} failed: ${res.status}`);
  const payload = await res.json();
  const arr = payload?.recent_data;
  if (!Array.isArray(arr) || !arr.length) return null;
  const last = arr[arr.length - 1];
  const close = last?.close;
  return typeof close === "number" ? close : null;
}

function pillClass(conf) {
  if (typeof conf !== "number") return "pill neutral";
  return conf >= 0.75 ? "pill good" : "pill neutral";
}

function buildPickRow(p) {
  const row = document.createElement("div");
  row.className = "row";

  const left = document.createElement("div");
  left.className = "left";

  const name = document.createElement("div");
  name.className = "name";
  name.textContent = (p.stock_name || p.symbol || "").toString();

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = (p.symbol || "").toString();

  left.appendChild(name);
  left.appendChild(meta);

  const pill = document.createElement("div");
  pill.className = pillClass(p.confidence);
  pill.textContent =
    typeof p.confidence === "number" ? `${(p.confidence * 100).toFixed(1)}%` : "—";

  row.appendChild(left);
  row.appendChild(pill);
  return row;
}

function buildPriceRow(symbol, close) {
  const row = document.createElement("div");
  row.className = "row";

  const left = document.createElement("div");
  left.className = "left";

  const name = document.createElement("div");
  name.className = "name";
  name.textContent = symbol;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = "Most recent close";

  left.appendChild(name);
  left.appendChild(meta);

  const pill = document.createElement("div");
  pill.className = "pill neutral";
  pill.textContent = close == null ? "n/a" : `$${close.toFixed(2)}`;

  row.appendChild(left);
  row.appendChild(pill);
  return row;
}

async function renderPredictions(payload) {
  const preds = payload?.predictions || [];
  stockBox.innerHTML = "";
  priceBox.innerHTML = "";

  if (!preds.length) {
    const d = document.createElement("div");
    d.className = "error";
    d.textContent = "No predictions returned.";
    stockBox.appendChild(d);
    return;
  }

  // Left: picks
  for (const p of preds) stockBox.appendChild(buildPickRow(p));

  // Right: actual numeric prices (latest close) for the same symbols.
  // Keep it fast: fetch sequentially; 10 requests is fine for the demo.
  for (const p of preds) {
    const sym = (p.symbol || "").toString();
    if (!sym) continue;
    try {
      const close = await fetchLatestClose(sym);
      priceBox.appendChild(buildPriceRow(sym, close));
    } catch {
      priceBox.appendChild(buildPriceRow(sym, null));
    }
  }
}

async function refresh() {
  try {
    if (refreshBtn) refreshBtn.disabled = true;
    const payload = await fetchPredictions();
    await renderPredictions(payload);
  } catch (e) {
    stockBox.innerHTML = "";
    const d = document.createElement("div");
    d.className = "error";
    d.textContent = `Error loading predictions: ${e.message}`;
    stockBox.appendChild(d);
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function updatePercentage() {
  const value = parseFloat(updateSlider.value).toFixed(2);
  percentDisplay.textContent = `${value}%`;
}

settingsBtn.addEventListener("click", () => {
  settingsPopup.classList.toggle("hidden");
});

updateSlider.addEventListener("input", updatePercentage);

document.addEventListener("click", (event) => {
  const clickedInsidePopup = settingsPopup.contains(event.target);
  const clickedGear = settingsBtn.contains(event.target);

  if (!clickedInsidePopup && !clickedGear) {
    settingsPopup.classList.add("hidden");
  }
});

updatePercentage();

if (refreshBtn) refreshBtn.addEventListener("click", refresh);

// Initial load for demo screenshot.
refresh();
