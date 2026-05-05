const settingsBtn = document.getElementById("settingsBtn");
const settingsPopup = document.getElementById("settingsPopup");
const updateSlider = document.getElementById("updateSlider");
const percentDisplay = document.getElementById("percentDisplay");
const stockBox = document.getElementById("stock-box");
const priceBox = document.getElementById("price-box");
const refreshBtn = document.getElementById("refreshBtn");

const buySymbol = document.getElementById("buySymbol");
const buyQty = document.getElementById("buyQty");
const buyPrice = document.getElementById("buyPrice");
const buyBtn = document.getElementById("buyBtn");
const toast = document.getElementById("toast");

const API_BASE = ""; // Use relative URL (works for both local and ngrok)

let selectedSymbol = null;
let selectedRowEl = null;

function showToast(msg) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast._t);
  showToast._t = window.setTimeout(() => toast.classList.add("hidden"), 2500);
}

function parseMoney(s) {
  if (s == null) return null;
  const t = String(s).replace(/[$,\s]/g, "").trim();
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

async function fetchPredictions() {
  // Expect API contract: { predictions: [{symbol, stock_name, confidence, horizon, model_version, as_of?}], count }
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
  row.className = "row selectable";

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

  row.addEventListener("click", async () => {
    const sym = (p.symbol || "").toString().trim().toUpperCase();
    if (!sym) return;

    if (selectedRowEl) selectedRowEl.classList.remove("selected");
    selectedRowEl = row;
    selectedRowEl.classList.add("selected");

    selectedSymbol = sym;
    if (buySymbol) buySymbol.value = sym;

    try {
      const close = await fetchLatestClose(sym);
      if (close != null && buyPrice) buyPrice.value = close.toFixed(2);
    } catch {
      // ignore
    }

    showToast(`Selected ${sym}`);
  });

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

  if (selectedRowEl) selectedRowEl.classList.remove("selected");
  selectedRowEl = null;

  if (!preds.length) {
    const d = document.createElement("div");
    d.className = "error";
    d.textContent = "No predictions returned.";
    stockBox.appendChild(d);
    return;
  }

  // Left: picks (clickable)
  for (const p of preds) stockBox.appendChild(buildPickRow(p));

  // Right: numeric prices (latest close) for same symbols.
  // Do this in parallel so the panel feels instant during the demo.
  const syms = preds
    .map((p) => (p.symbol || "").toString().trim().toUpperCase())
    .filter((s) => s);

  const results = await Promise.all(
    syms.map(async (sym) => {
      try {
        const close = await fetchLatestClose(sym);
        return { sym, close };
      } catch {
        return { sym, close: null };
      }
    })
  );

  for (const r of results) {
    priceBox.appendChild(buildPriceRow(r.sym, r.close));
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

async function onBuy() {
  const sym = (buySymbol?.value || selectedSymbol || "").toString().trim().toUpperCase();
  const qty = Number(buyQty?.value || 1);

  if (!sym) return showToast("Enter or select a symbol");
  if (!Number.isFinite(qty) || qty <= 0) return showToast("Quantity must be > 0");

  let price = parseMoney(buyPrice?.value);
  if (price == null) {
    try {
      const close = await fetchLatestClose(sym);
      if (close != null) {
        price = close;
        if (buyPrice) buyPrice.value = close.toFixed(2);
      }
    } catch {
      // ignore
    }
  }

  if (price == null) return showToast("Price missing (click a stock or type a price)");

  showToast(`Paper buy: ${qty} ${sym} @ $${price.toFixed(2)} (demo only)`);
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
if (buyBtn) buyBtn.addEventListener("click", onBuy);

// Initial load for demo screenshot.
refresh();
