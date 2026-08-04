/* mcufit web — runs the mcufit Python package in the browser via Pyodide.
 * The wheel is built from this repo at deploy time and served from this
 * site's own origin (web/wheels/), so page and package can never drift.
 * The model file is analyzed in the in-browser filesystem; nothing is
 * uploaded anywhere. */

const el = (id) => document.getElementById(id);
const fmt = (bytes) => {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  if (bytes >= 1024) return Math.round(bytes / 1024) + " KB";
  return bytes + " B";
};

const PY_SETUP = `
import json
from pathlib import Path
from mcufit.parsing.tflite_parser import TFLiteModelParser
from mcufit.estimation.greedy import GreedyLifetimeEstimator
from mcufit.boards.yaml_repo import YamlBoardRepository
from mcufit.analysis.fit_checker import FitChecker

_repo = YamlBoardRepository()
_checker = FitChecker(estimator=GreedyLifetimeEstimator(), boards=_repo)
_parser = TFLiteModelParser()

def list_boards():
    return json.dumps([
        {"id": b.id, "name": b.name, "chip": b.chip, "vendor": b.vendor,
         "usable_sram": b.usable_sram_bytes, "flash": b.flash_bytes,
         "psram": b.psram_bytes}
        for b in _repo.list()
    ])

class _FixedEstimator:
    """Feeds browser-measured (WASM TFLM) numbers through the verdict logic."""

    def __init__(self, nonpersistent, persistent):
        self._np = nonpersistent
        self._p = persistent

    def estimate(self, model):
        from mcufit.domain.report import MemoryEstimate
        return MemoryEstimate(
            peak_activation_bytes=self._np, overhead_bytes=self._p,
            margin_bytes=0, peak_layer_index=-1, method="tflm-wasm-measurement",
        )

def analyze(path, board_id, measured_np=None, measured_p=None):
    model = _parser.parse(Path(path))
    checker = _checker
    if measured_np is not None:
        checker = FitChecker(
            estimator=_FixedEstimator(int(measured_np), int(measured_p or 0)),
            boards=_repo,
        )
    rows = []
    selected = None
    for board in _repo.list():
        report = checker.check(model, board)
        row = {
            "board_id": board.id, "board_name": board.name,
            "usable_sram": board.usable_sram_bytes,
            "arena": report.estimate.total_arena_bytes,
            "fits": report.fits,
        }
        rows.append(row)
        if board.id == board_id:
            est = report.estimate
            peak_layer = model.layers[est.peak_layer_index] if model.layers else None
            selected = {
                "fits": report.fits, "fits_ram": report.fits_ram,
                "fits_flash": report.fits_flash,
                "arena": est.total_arena_bytes,
                "peak_activation": est.peak_activation_bytes,
                "peak_layer_index": est.peak_layer_index,
                "peak_layer_op": peak_layer.op_name if peak_layer else "",
                "flash_needed": report.flash_needed_bytes,
                "usable_sram": board.usable_sram_bytes,
                "board_flash": board.flash_bytes,
                "board_name": board.name,
                "ram_utilization": report.ram_utilization,
                "flash_utilization": report.flash_utilization,
                "suggestions": [s.text for s in report.suggestions],
            }
    meta = {
        "quantization": model.quantization.value,
        "layers": len(model.layers),
        "file_size": model.file_size_bytes,
        "measured": measured_np is not None,
    }
    return json.dumps({"selected": selected, "rows": rows, "model": meta})
`;

let pyodide = null;
let currentFile = null; // { name, bytes }
let currentMeasured = null; // { np, p } from the WASM TFLM run, or null

let tflmFactoryPromise = null;
function loadTflm() {
  if (!tflmFactoryPromise) {
    tflmFactoryPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "wasm/tflm.js";
      s.onload = () => resolve(window.createTflmModule);
      s.onerror = () => reject(new Error("could not load wasm/tflm.js"));
      document.head.appendChild(s);
    });
  }
  return tflmFactoryPromise;
}

/* Run the model through the real TFLM interpreter compiled to WebAssembly
 * and read its recorded arena allocations. Returns null on any failure —
 * the static estimate then stands. */
async function measureArena(bytes) {
  try {
    const factory = await loadTflm();
    const lines = [];
    const mod = await factory({
      print: (t) => lines.push(t),
      printErr: (t) => lines.push(t),
    });
    mod.FS.writeFile("/model.tflite", bytes);
    try {
      mod.callMain(["/model.tflite"]);
    } catch (e) {
      // emscripten raises ExitStatus on clean exit; output is already captured
    }
    const section = lines.join("\n").split("[[ Table ]]: Arena")[1] || "";
    const np = /NonPersistent\s*\|\s*(\d+)/.exec(section);
    const p = /\bPersistent\s*\|\s*(\d+)/.exec(section);
    return np ? { np: +np[1], p: p ? +p[1] : 0 } : null;
  } catch (err) {
    console.warn("wasm measurement unavailable:", err);
    return null;
  }
}

async function boot() {
  try {
    el("status-detail").textContent = "(~8 MB, first visit only)";
    pyodide = await loadPyodide();
    el("status").firstChild.textContent = "Installing mcufit… ";
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(["tflite", "pyyaml", "numpy"]);
    const manifest = await (await fetch("wheels/manifest.json", { cache: "no-cache" })).json();
    await micropip.install(new URL(`wheels/${manifest.wheel}`, location.href).href, { deps: false });
    pyodide.runPython(PY_SETUP);
    populateBoards();
    el("status").hidden = true;
    el("app").hidden = false;
  } catch (err) {
    el("status").classList.add("error");
    el("status").textContent =
      "Failed to load the analysis engine — please refresh, or use the CLI: pip install mcufit. (" + err + ")";
  }
}

function populateBoards() {
  const boards = JSON.parse(pyodide.runPython("list_boards()"));
  const select = el("board-select");
  el("board-label").textContent = `Target board (${boards.length} in database)`;

  const groups = new Map();
  for (const b of boards) {
    if (!groups.has(b.vendor)) groups.set(b.vendor, []);
    groups.get(b.vendor).push(b);
  }
  const vendors = [...groups.keys()].sort((a, b) =>
    a === "Other" ? 1 : b === "Other" ? -1 : a.localeCompare(b));

  for (const vendor of vendors) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = vendor;
    for (const b of groups.get(vendor).sort((a, b) => a.name.localeCompare(b.name))) {
      const opt = document.createElement("option");
      opt.value = b.id;
      opt.textContent = `${b.name} — ${fmt(b.usable_sram)} SRAM`;
      optgroup.appendChild(opt);
    }
    select.appendChild(optgroup);
  }
  select.value = "esp32-s3";
}

function analyze() {
  if (!currentFile || !pyodide) return;
  pyodide.FS.writeFile("/tmp/model.tflite", currentFile.bytes);
  const board = el("board-select").value;
  const m = currentMeasured;
  let result;
  try {
    const analyzeFn = pyodide.globals.get("analyze");
    result = JSON.parse(analyzeFn("/tmp/model.tflite", board, m ? m.np : null, m ? m.p : null));
    analyzeFn.destroy();
  } catch (err) {
    alert("Could not analyze this file — is it a valid .tflite model?\n\n" + err);
    return;
  }
  render(result);
}

function render({ selected: s, rows, model }) {
  el("result").hidden = false;

  const verdict = s.fits
    ? '<span class="verdict-line fits">✅ FITS</span>'
    : '<span class="verdict-line no-fit">❌ WON\'T FIT</span>';
  el("verdict-head").innerHTML =
    `<strong>${currentFile.name}</strong> (${model.quantization}, ${model.layers} layers, ${fmt(model.file_size)}) ` +
    `on <strong>${s.board_name}</strong>${verdict}`;

  setMeter("ram", s.ram_utilization, s.fits_ram,
    `~${fmt(s.arena)} arena / ${fmt(s.usable_sram)} (${Math.min(999, Math.round(s.ram_utilization * 100))}%)`);
  setMeter("flash", s.flash_utilization, s.fits_flash,
    `${fmt(s.flash_needed)} total / ${fmt(s.board_flash)} (${Math.min(999, Math.round(s.flash_utilization * 100))}%)`);

  el("peak-info").textContent =
    `Peak memory moment: layer ${s.peak_layer_index} (${s.peak_layer_op}) — ${fmt(s.peak_activation)} of simultaneously-live tensors.`;

  const ul = el("suggestions");
  ul.innerHTML = "";
  for (const text of s.suggestions) {
    const li = document.createElement("li");
    li.textContent = text;
    ul.appendChild(li);
  }

  const note = el("method-note");
  if (model.measured) {
    note.classList.add("measured");
    note.innerHTML =
      "<strong>✓ Measured by the real TFLite Micro runtime</strong> (compiled to " +
      "WebAssembly, 32-bit — the same allocation numbers a device reports). " +
      'No estimate involved. CLI equivalent: <code>mcufit check model.tflite -b board --exact</code>';
  } else {
    note.classList.remove("measured");
    note.innerHTML =
      "<strong>⚠️ Fast estimate</strong> (static analysis; measuring with the real " +
      "runtime in the background…). For exact numbers in the terminal: " +
      "<code>pip install mcufit && mcufit setup-exact && mcufit check model.tflite -b esp32-s3 --exact</code>";
  }

  const tbody = el("matrix").querySelector("tbody");
  tbody.innerHTML = "";
  const tilde = model.measured ? "" : "~";
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.board_id === el("board-select").value) tr.classList.add("highlight");
    tr.innerHTML =
      `<td>${row.board_name}</td>` +
      `<td class="num">${fmt(row.usable_sram)}</td>` +
      `<td class="num">${tilde}${fmt(row.arena)}</td>` +
      `<td>${row.fits ? '<span class="fits">✅ fits</span>' : '<span class="no-fit">❌ no</span>'}</td>`;
    tbody.appendChild(tr);
  }
}

function setMeter(kind, ratio, ok, text) {
  const bar = el(`${kind}-bar`);
  bar.style.width = Math.min(100, ratio * 100) + "%";
  bar.className = "meter-fill " + (ok ? "ok" : "over");
  el(`${kind}-text`).textContent = text;
}

function acceptFile(file) {
  file.arrayBuffer().then((buf) => setModel(file.name, new Uint8Array(buf)));
}

function setModel(name, bytes) {
  const model = { name, bytes };
  currentFile = model;
  currentMeasured = null;
  analyze(); // instant static verdict first
  measureArena(bytes).then((m) => {
    if (m && currentFile === model) {
      currentMeasured = m;
      analyze(); // re-render with measured numbers
    }
  });
}

// --- wiring ---
const dropzone = el("dropzone");
dropzone.addEventListener("click", () => el("file-input").click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") el("file-input").click(); });
el("file-input").addEventListener("change", (e) => { if (e.target.files[0]) acceptFile(e.target.files[0]); });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) acceptFile(e.dataTransfer.files[0]);
});
el("board-select").addEventListener("change", analyze);
el("example-btn").addEventListener("click", async () => {
  const resp = await fetch("examples/person_detect.tflite");
  const buf = await resp.arrayBuffer();
  setModel("person_detect.tflite", new Uint8Array(buf));
});

boot();
