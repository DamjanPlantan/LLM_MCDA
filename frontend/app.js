// ============================================================
// KLJUČNE TOČKE (na kratko):
// - "Ustvari povzetek" pokliče backend, doda alternativo v tabelo in MCDA matriko.
// - "Naredi odločitve" pošlje MCDA matriko na backend in vrne TOPSIS + PROMETHEE.
// - Test se shrani v zgodovino šele po MCDA.
// ============================================================

function el(id) {
  return document.getElementById(id);
}

// ------------------------------------------------------------
// Pomožne funkcije za format števil
// ------------------------------------------------------------
function format1(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "";
  return n.toFixed(1);
}
function format2(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "";
  return n.toFixed(2);
}
function format3(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "";
  return n.toFixed(3);
}

function buildOptions1to5(selected) {
  // Naredi <option> elemente od 1 do 5 in označi izbranega.
  // Začetna vrednost je PRAZNA (brez izbire), dokler uporabnik ne izbere.
  const s = (selected === undefined || selected === null || selected === "") ? "" : String(selected);
  let html = "";

  // Prazen placeholder (uporabnik mora izbrati)
  const selEmpty = (s === "") ? " selected" : "";
  html += `<option value=""${selEmpty}></option>`;

  for (let i = 1; i <= 5; i++) {
    const sel = (String(i) === s) ? " selected" : "";
    html += `<option value="${i}"${sel}>${i}</option>`;
  }
  return html;
}

function parseDecimal(value) {
  // Dovolimo tudi vejico kot decimalno ločilo (npr. "0,2")
  const s = String(value || "").replace(",", ".").trim();
  const n = parseFloat(s);
  return Number.isNaN(n) ? 0 : n;
}

// ------------------------------------------------------------
// Trenutna seja (v brskalniku)
// ------------------------------------------------------------
let currentTest = {
  test_id: "",
input_text: "",
  input_chars: 0,
  weights: {
    // 2 decimalki + vsota 1.00
    time_s: 0.17,
    ram_peak_mb: 0.17,
    ram_avg_mb: 0.17,
    output_pct: 0.17,
    quality: 0.16,
    hallucinations: 0.16
  },
  alternatives: [],
  topsis: [],
  promethee: []
};

function showBusy(flag) {
  if (flag) el("busyBox").classList.remove("hidden");
  else el("busyBox").classList.add("hidden");
}

function alertError(msg) {
  window.alert(msg);
}

// ------------------------------------------------------------
// Modeli
// ------------------------------------------------------------
async function loadModels() {
  const sel = el("modelSelect");
  sel.innerHTML = "";

  try {
    const r = await fetch("/api/models");
    const data = await r.json();

    if (!data.ok) {
      alertError(data.error || "Napaka pri nalaganju modelov.");
      return;
    }

    data.models.forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });

    if (data.models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(ni modelov)";
      sel.appendChild(opt);
    }
  } catch (e) {
    alertError("Ollama ni dosegljiv ali backend ne deluje.\n\n" + e);
  }
}

// ------------------------------------------------------------
// Render: Rezultati testov
// ------------------------------------------------------------
function renderResultsTable() {
  const tb = el("resultsTbody");
  tb.innerHTML = "";

  currentTest.alternatives.forEach(alt => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${alt.alternative}</td>
      <td>${format1(alt.time_s)}</td>
      <td>${format3(alt.ram_peak_mb)}</td>
      <td>${format3(alt.ram_avg_mb)}</td>
      <td>${format2(alt.output_pct)}</td>
      <td>${currentTest.input_chars}</td>
      <td>${alt.output_chars}</td>
      <td>${alt.token_in ?? ""}</td>
      <td>${alt.token_out ?? ""}</td>
    `;
    tb.appendChild(tr);
  });
}

// ------------------------------------------------------------
// Render: MCDA matrika
// ------------------------------------------------------------
function renderMcdaMatrix() {
  const tb = el("mcdaTbody");

  // POMEMBNO:
  // Če uporabnik spremeni uteži, so te uteži najprej samo v poljih (v UI).
  // Ko dodamo novo alternativo, matriko ponovno izrišemo.
  // Zato MORAMO najprej prebrati uteži iz UI in jih shraniti v currentTest.weights,
  // sicer bi se uteži vrnile na začetne vrednosti.
  if (el("w_time_s")) {
    readWeightsFromUi();
  }

  // Enako velja za ročne kriterije (kvaliteta in halucinacije)
  // Če so dropdowni že na strani, preberemo vrednosti in jih shranimo.
  if (el("q_0") !== null || el("h_0") !== null) {
    readManualCriteriaFromUi();
  }

  tb.innerHTML = "";

  // 1) UTEŽI
  const trW = document.createElement("tr");
  trW.innerHTML = `
    <td><b>UTEŽI</b></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_time_s" value="${format2(currentTest.weights.time_s)}"></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_ram_peak_mb" value="${format2(currentTest.weights.ram_peak_mb)}"></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_ram_avg_mb" value="${format2(currentTest.weights.ram_avg_mb)}"></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_output_pct" value="${format2(currentTest.weights.output_pct)}"></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_quality" value="${format2(currentTest.weights.quality)}"></td>
    <td><input class="weight-input" type="text" inputmode="decimal" id="w_hallucinations" value="${format2(currentTest.weights.hallucinations)}"></td>
  `;
  tb.appendChild(trW);

  // 2) Alternative
  currentTest.alternatives.forEach((alt, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${alt.alternative}</td>
      <td><input type="text" value="${format1(alt.time_s)}" readonly></td>
      <td><input type="text" value="${format3(alt.ram_peak_mb)}" readonly></td>
      <td><input type="text" value="${format3(alt.ram_avg_mb)}" readonly></td>
      <td><input type="text" value="${format2(alt.output_pct)}" readonly></td>
      <td><select id="q_${idx}">${buildOptions1to5(alt.quality)}</select></td>
      <td><select id="h_${idx}">${buildOptions1to5(alt.hallucinations)}</select></td>
    `;
    tb.appendChild(tr);
  });
}

// ------------------------------------------------------------
// Render: TOPSIS / PROMETHEE
// ------------------------------------------------------------
function renderTopsis(rows) {
  const tb = el("topsisTbody");
  tb.innerHTML = "";

  // Sort po rangu (1, 2, 3, ...)
  const sorted = [...rows].sort((a, b) => Number(a.Rank) - Number(b.Rank));

  sorted.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.Alternative}</td>
      <td>${r.D_plus}</td>
      <td>${r.D_minus}</td>
      <td>${r.C_star}</td>
      <td>${r.Rank}</td>
    `;
    tb.appendChild(tr);
  });
}

function renderPromethee(rows) {
  const tb = el("promTbody");
  tb.innerHTML = "";

  // Sort po rangu (1, 2, 3, ...)
  const sorted = [...rows].sort((a, b) => Number(a.Rank) - Number(b.Rank));

  sorted.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.Alternative}</td>
      <td>${r.Phi_plus}</td>
      <td>${r.Phi_minus}</td>
      <td>${r.Phi}</td>
      <td>${r.Rank}</td>
    `;
    tb.appendChild(tr);
  });
}


// ------------------------------------------------------------
// Branje uteži + ročnih kriterijev
// ------------------------------------------------------------
function readWeightsFromUi() {
  currentTest.weights.time_s = parseDecimal(el("w_time_s").value);
  currentTest.weights.ram_peak_mb = parseDecimal(el("w_ram_peak_mb").value);
  currentTest.weights.ram_avg_mb = parseDecimal(el("w_ram_avg_mb").value);
  currentTest.weights.output_pct = parseDecimal(el("w_output_pct").value);
  currentTest.weights.quality = parseDecimal(el("w_quality").value);
  currentTest.weights.hallucinations = parseDecimal(el("w_hallucinations").value);
}

function readManualCriteriaFromUi() {
  // Prebere ročne kriterije (kvaliteta, halucinacije) iz UI in jih shrani v podatke.
  // POMEMBNO:
  // - Ko dodamo novo alternativo, UI še nima q_1/h_1, zato moramo preveriti, ali element obstaja.
  // - Če je dropdown prazen, shranimo null.
  currentTest.alternatives.forEach((alt, idx) => {
    const qEl = el("q_" + idx);
    const hEl = el("h_" + idx);

    if (qEl !== null) {
      const v = String(qEl.value || "").trim();
      alt.quality = (v === "") ? null : Number(v);
    }

    if (hEl !== null) {
      const v = String(hEl.value || "").trim();
      alt.hallucinations = (v === "") ? null : Number(v);
    }
  });
}


// ------------------------------------------------------------
// Ustvari povzetek
// ------------------------------------------------------------
async function doSummarize() {
  // Vedno najprej pobrišemo polje povzetka (da se vidi, da teče nov test)
  el("outputText").value = "";

  // Preden dodamo nov model, shranimo trenutne ročne kriterije in uteži (da se ne izgubijo)
  if (el("w_time_s") !== null) {
    readWeightsFromUi();
  }
  if (el("q_0") !== null || el("h_0") !== null) {
    readManualCriteriaFromUi();
  }

  const model = el("modelSelect").value;
  const inputText = el("inputText").value.trim();

  if (!model) return alertError("Najprej izberi model.");
  if (!inputText) return alertError("Najprej vpiši vhodno besedilo.");

  // Isti model ne sme biti 2x v matriki
  const already = currentTest.alternatives.some(a => a.alternative === model);
  if (already) return alertError("Ta model je že dodan v matriko. Isti model ne sme biti 2x.");

  currentTest.input_text = inputText;
  currentTest.input_chars = inputText.length;

  showBusy(true);

  try {
    const r = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model, input_text: inputText })
    });

    const data = await r.json();
    if (!data.ok) {
      showBusy(false);
      return alertError(data.error || "Napaka pri povzetku.");
    }

    // povzetek desno
    el("outputText").value = data.summary;

    // alternativa = ime modela
    const alt = {
      alternative: data.model,
      // Shranimo tudi vhodni tekst (input), da ga lahko izvozimo skupaj s povzetkom
      input_text: currentTest.input_text,
      summary: data.summary,

      // read-only kriteriji iz testa
      time_s: data.time_s,
      ram_peak_mb: data.ram_peak_mb,
      ram_avg_mb: data.ram_avg_mb,
      output_pct: data.output_pct,

      output_chars: data.output_chars,
      token_in: data.token_in,
      token_out: data.token_out,

      // ročni kriteriji
      quality: null,
      hallucinations: null
    };

    currentTest.alternatives.push(alt);

    renderResultsTable();
    renderMcdaMatrix();

  } catch (e) {
    alertError("Napaka pri klicu backenda:\n\n" + e);
  } finally {
    showBusy(false);
  }
}

// ------------------------------------------------------------
// Naredi odločitve
// ------------------------------------------------------------
async function doMcda() {
  if (currentTest.alternatives.length < 2) {
    return alertError("Za izvedbo MCDA sta potrebni najmanj 2 alternativi.");
  }

  readWeightsFromUi();
  readManualCriteriaFromUi();

  const now = new Date();
  currentTest.test_id = "TEST_" + now.getTime();
  try {
    const r = await fetch("/api/mcda", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test: currentTest, save_to_history: true })
    });

    const data = await r.json();
    if (!data.ok) {
      return alertError(data.error || "Napaka pri MCDA.");
    }

    currentTest.topsis = data.topsis || [];
    currentTest.promethee = data.promethee || [];

    renderTopsis(currentTest.topsis);
    renderPromethee(currentTest.promethee);

  } catch (e) {
    alertError("Napaka pri MCDA:\n\n" + e);
  }
}

// ------------------------------------------------------------
// Počisti okno (samo trenutna seja)
// ------------------------------------------------------------
function clearWindow() {
  currentTest = {
    test_id: "",
input_text: "",
    input_chars: 0,
    weights: {
      time_s: 0.17,
      ram_peak_mb: 0.17,
      ram_avg_mb: 0.17,
      output_pct: 0.17,
      quality: 0.16,
      hallucinations: 0.16
    },
    alternatives: [],
    topsis: [],
    promethee: []
  };

  el("inputText").value = "";
  el("outputText").value = "";

  renderResultsTable();
  renderMcdaMatrix();
  renderTopsis([]);
  renderPromethee([]);
}

// ------------------------------------------------------------
// Izbriši zgodovino
// ------------------------------------------------------------
async function clearHistory() {
  if (!confirm("Ali res želiš izbrisati vso zgodovino testov?")) return;

  try {
    const r = await fetch("/api/history/clear", { method: "POST" });
    const data = await r.json();
    if (!data.ok) return alertError("Napaka pri brisanju zgodovine.");
    alert("Zgodovina je izbrisana.");
  } catch (e) {
    alertError("Napaka:\n\n" + e);
  }
}

// ------------------------------------------------------------
// Izvoz
// ------------------------------------------------------------
function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

async function exportCurrent() {
  if (!currentTest.topsis.length && !currentTest.promethee.length) {
    return alertError("Najprej izvedi MCDA (Naredi odločitve).");
  }

  try {
    const r = await fetch("/api/export/current", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test: currentTest })
    });

    if (!r.ok) return alertError("Napaka pri izvozu.");
    const blob = await r.blob();
    downloadBlob(blob, "current_test.xlsx");
  } catch (e) {
    alertError("Napaka:\n\n" + e);
  }
}

async function exportAll() {
  try {
    const r = await fetch("/api/export/all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test: currentTest })
    });

    if (!r.ok) return alertError("Napaka pri izvozu.");
    const blob = await r.blob();
    downloadBlob(blob, "all_tests.xlsx");
  } catch (e) {
    alertError("Napaka:\n\n" + e);
  }
}

// ------------------------------------------------------------
// INIT
// ------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  el("btnSummarize").addEventListener("click", () => {
    // Pobriši povzetek takoj ob kliku (še preden gre zahteva na backend)
    el("outputText").value = "";
    doSummarize();
  });
  el("btnMcda").addEventListener("click", doMcda);
  el("btnClear").addEventListener("click", clearWindow);
  el("btnClearHistory").addEventListener("click", clearHistory);
  el("btnExportCurrent").addEventListener("click", exportCurrent);
  el("btnExportAll").addEventListener("click", exportAll);
  el("btnRefreshModels").addEventListener("click", loadModels);

  await loadModels();

  renderResultsTable();
  renderMcdaMatrix();
  renderTopsis([]);
  renderPromethee([]);
});
