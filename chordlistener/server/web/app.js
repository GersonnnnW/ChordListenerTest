"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.has("embed")) document.documentElement.classList.add("embed");

const state = {
  id: null,
  result: null,
  source: null,
  transpose: 0,
  arrangement: "piano",
  objectUrl: null,
  input: null, // what was analysed, so it can be re-run with other settings
  tempo: "auto",
  sheetKey: null, // what the sheet view currently shows
  osmd: null,
  beatEls: [],
  barEls: [],
  lastBeat: -1,
  lastChord: -2,
};

// ---------------------------------------------------------------- helpers

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Index of the last entry in sorted `values` that is <= x, or -1. */
function lastAtOrBefore(values, x) {
  let lo = 0, hi = values.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] <= x) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found;
}

function show(section) {
  for (const id of ["start", "working", "result"]) $(id).hidden = id !== section;
}

function fail(message) {
  show("start");
  $("error").textContent = message;
  $("error").hidden = false;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  const isJson = (response.headers.get("content-type") || "").includes("json");
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok && response.status !== 202) {
    const message = (body && body.error && body.error.message) || "Something went wrong. Please try again.";
    throw new Error(message);
  }
  return body;
}

// ---------------------------------------------------------------- search

let searchTimer = 0;
let searchSeq = 0;

$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = $("search").value.trim();
  if (query.length < 2) { $("results").replaceChildren(); return; }
  searchTimer = setTimeout(() => runSearch(query), 280);
});

async function runSearch(query) {
  const seq = ++searchSeq;
  let data;
  try {
    data = await api(`/v1/songs/search?limit=6&q=${encodeURIComponent(query)}`);
  } catch (error) {
    if (seq === searchSeq) $("results").replaceChildren(el("li", "empty", error.message));
    return;
  }
  if (seq !== searchSeq) return; // a newer search has started
  const list = $("results");
  list.replaceChildren();
  if (!data.results.length) {
    list.append(el("li", "empty", "No songs found. Try the artist's name too."));
    return;
  }
  for (const track of data.results) {
    const button = el("button");
    button.type = "button";
    const art = el("img");
    art.alt = "";
    art.loading = "lazy";
    if (track.artwork_url) art.src = track.artwork_url;
    const text = el("div");
    text.append(el("div", "t", track.title), el("div", "a", track.artist));
    button.append(art, text);
    button.addEventListener("click", () => analyzeTrack(track));
    const item = el("li");
    item.append(button);
    list.append(item);
  }
}

// ---------------------------------------------------------------- upload

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => { if ($("file").files[0]) analyzeFile($("file").files[0]); });
for (const type of ["dragenter", "dragover"]) {
  drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add("over"); });
}
for (const type of ["dragleave", "drop"]) {
  drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove("over"); });
}
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) analyzeFile(file);
});
// Dropping a file anywhere else should not navigate away from the page.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

function analyzeFile(file) {
  releaseAudio();
  state.objectUrl = URL.createObjectURL(file);
  state.input = { file, name: file.name, audioUrl: state.objectUrl };
  state.tempo = "auto";
  submit();
}

function analyzeTrack(track) {
  releaseAudio();
  state.input = { track, name: `${track.title} by ${track.artist}`, audioUrl: track.preview_url };
  state.tempo = "auto";
  submit();
}

async function submit() {
  const input = state.input;
  const form = new FormData();
  if (input.file) {
    form.append("file", input.file);
    form.append("title", input.file.name.replace(/\.[^.]+$/, ""));
  } else {
    form.append("track_id", input.track.track_id);
  }
  form.append("tempo", state.tempo);
  $("error").hidden = true;
  $("working-name").textContent = input.name;
  show("working");
  try {
    let body = await api("/v1/transcriptions", { method: "POST", body: form });
    while (body.status === "processing") {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      body = await api(`/v1/transcriptions/${body.id}`);
    }
    if (body.status !== "completed") throw new Error((body.error && body.error.message) || "The analysis failed.");
    present(body, input.audioUrl);
  } catch (error) {
    fail(error.message);
  } finally {
    $("file").value = "";
  }
}

// Whether a song is "70 bpm" or "140 bpm" is a judgement call; let people flip it.
function retime(direction) {
  const order = ["half", "auto", "double"];
  const next = order[order.indexOf(state.tempo) + direction];
  if (!next || !state.input) return;
  $("audio").pause();
  state.tempo = next;
  submit();
}
$("half").addEventListener("click", () => retime(-1));
$("double").addEventListener("click", () => retime(1));

function releaseAudio() {
  const audio = $("audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
}

// ---------------------------------------------------------------- result

function present(body, audioUrl) {
  state.id = body.id;
  state.source = body.source;
  state.transpose = 0;
  state.sheetKey = null;
  $("sheet").replaceChildren();
  $("sheet-status").textContent = "";

  const source = body.source;
  $("song-title").textContent = source.title || (source.filename || "Untitled").replace(/\.[^.]+$/, "");
  $("song-artist").textContent = source.artist || "";
  $("artwork").hidden = !source.artwork_url;
  if (source.artwork_url) $("artwork").src = source.artwork_url;
  $("clip-note").hidden = source.type !== "catalog";

  const player = $("player");
  const url = audioUrl || source.preview_url || "";
  player.classList.toggle("silent", !url);
  if (url && $("audio").getAttribute("src") !== url) $("audio").src = url;
  $("half").hidden = $("double").hidden = !state.input;
  $("half").disabled = state.tempo === "half";
  $("double").disabled = state.tempo === "double";
  setPlaying(false);
  $("seek").value = 0;

  applyResult(body.result);
  selectTab("chords");
  show("result");
  history.replaceState(null, "", `?${params.has("embed") ? "embed=1&" : ""}t=${encodeURIComponent(body.id)}`);
  window.scrollTo({ top: 0 });
}

function applyResult(result) {
  state.result = result;
  $("fact-key").textContent = result.key.name;
  $("fact-tempo").textContent = `${Math.round(result.tempo_bpm)} bpm`;
  $("fact-time").textContent = result.time_signature;
  $("semitones").textContent = (state.transpose > 0 ? "+" : "") + state.transpose;
  $("seek").max = result.duration;
  renderChart(result);
  state.lastBeat = -1;
  state.lastChord = -2;
  sync();

  const base = `/v1/transcriptions/${state.id}`;
  const shift = `transpose=${state.transpose}`;
  $("dl-xml").href = `${base}/musicxml?arrangement=${state.arrangement}&${shift}&download=true`;
  $("dl-midi").href = `${base}/midi?${shift}`;
  $("dl-chart").href = `${base}/chart?${shift}&download=true`;
  $("dl-json").href = `${base}?${shift}`;
}

function renderChart(result) {
  const perBar = parseInt(result.time_signature, 10) || 4;
  const bars = Math.max(1, Math.floor((result.beats.length - 1) / perBar));
  const starts = new Map(); // global beat index -> chord symbol
  for (const chord of result.chords) starts.set((chord.bar - 1) * perBar + (chord.beat - 1), chord.chord);

  const chart = $("chart");
  chart.replaceChildren();
  state.beatEls = [];
  state.barEls = [];
  let carried = "";
  for (let bar = 0; bar < bars; bar++) {
    const cell = el("button", "bar");
    cell.type = "button";
    cell.append(el("span", "n", String(bar + 1)));
    let changes = 0;
    for (let beat = 0; beat < perBar; beat++) {
      const index = bar * perBar + beat;
      const span = el("span", "beat");
      let symbol = starts.get(index);
      if (symbol !== undefined) { carried = symbol; changes++; }
      else if (beat === 0) symbol = carried; // remind the reader what is still sounding
      if (symbol === undefined || symbol === "") span.classList.add("held");
      else if (symbol === "N.C.") { span.classList.add("rest"); span.textContent = "–"; }
      else span.textContent = symbol;
      cell.append(span);
      state.beatEls.push(span);
    }
    if (changes > 2) cell.classList.add("dense");
    cell.addEventListener("click", () => seekTo(result.beats[bar * perBar]));
    cell.setAttribute("aria-label", `Bar ${bar + 1}`);
    chart.append(cell);
    state.barEls.push(cell);
  }
}

// ---------------------------------------------------------------- playback

const audio = $("audio");

function setPlaying(playing) {
  $("play-icon").setAttribute("d", playing ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z");
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
}

$("play").addEventListener("click", () => { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); });
audio.addEventListener("play", () => { setPlaying(true); requestAnimationFrame(tick); });
audio.addEventListener("pause", () => setPlaying(false));
audio.addEventListener("ended", () => setPlaying(false));
audio.addEventListener("timeupdate", sync);
audio.addEventListener("error", () => { if (audio.getAttribute("src")) $("player").classList.add("silent"); });
$("seek").addEventListener("input", () => seekTo(parseFloat($("seek").value)));

function seekTo(seconds) {
  if (!$("player").classList.contains("silent")) audio.currentTime = Math.max(0, seconds);
  sync(Math.max(0, seconds));
}

function tick() {
  sync();
  if (!audio.paused) requestAnimationFrame(tick);
}

function sync(at) {
  const result = state.result;
  if (!result) return;
  const time = typeof at === "number" ? at : audio.currentTime || 0;
  if (document.activeElement !== $("seek")) $("seek").value = time;
  $("clock").textContent = `${clock(time)} / ${clock(result.duration)}`;

  const beat = Math.min(lastAtOrBefore(result.beats, time), state.beatEls.length - 1);
  if (beat !== state.lastBeat) {
    const perBar = parseInt(result.time_signature, 10) || 4;
    const before = state.beatEls[state.lastBeat];
    if (before) before.classList.remove("on");
    const oldBar = state.barEls[Math.floor(state.lastBeat / perBar)];
    if (oldBar) oldBar.classList.remove("active");
    const now = state.beatEls[beat];
    if (now) now.classList.add("on");
    const bar = state.barEls[Math.floor(beat / perBar)];
    if (bar) {
      bar.classList.add("active");
      if (!audio.paused && !$("view-chords").hidden) bar.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    state.lastBeat = beat;
  }

  const chords = result.chords;
  let index = -1;
  for (let lo = 0, hi = chords.length - 1; lo <= hi; ) {
    const mid = (lo + hi) >> 1;
    if (chords[mid].start <= time) { index = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  if (index >= 0 && time >= chords[index].end) index = -1;
  if (index !== state.lastChord) {
    state.lastChord = index;
    const current = chords[index];
    $("now-chord").textContent = current ? current.chord : "–";
    const upcoming = chords.find((c) => c.start > time && c.chord !== (current && current.chord));
    $("next-chord").textContent = upcoming ? upcoming.chord : "–";
    drawKeys(current ? current.notes : [], current ? current.root : null);
  }
}

// A two-octave keyboard that lights up the notes of the current chord.
const PITCH = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const pitchClass = (name) => (PITCH[name[0]] + (name.match(/#/g) || []).length - (name.slice(1).match(/b/g) || []).length + 12) % 12;

function drawKeys(notes, root) {
  const lit = new Set(notes.map(pitchClass));
  const rootPc = root ? pitchClass(root) : -1;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 196 56");
  const whites = [0, 2, 4, 5, 7, 9, 11];
  const blacks = [[1, 0], [3, 1], [6, 3], [8, 4], [10, 5]];
  const key = (x, width, height, pc, black) => {
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", 1);
    rect.setAttribute("width", width); rect.setAttribute("height", height);
    rect.setAttribute("rx", 2);
    const on = lit.has(pc);
    rect.setAttribute("fill", on ? "var(--accent)" : black ? "var(--text)" : "var(--surface)");
    rect.setAttribute("fill-opacity", on && pc !== rootPc ? "0.6" : "1");
    rect.setAttribute("stroke", "var(--muted)");
    rect.setAttribute("stroke-width", "1");
    svg.append(rect);
  };
  for (let octave = 0; octave < 2; octave++) {
    whites.forEach((pc, i) => key(1 + (octave * 7 + i) * 13.85, 13.85, 54, pc, false));
  }
  for (let octave = 0; octave < 2; octave++) {
    blacks.forEach(([pc, slot]) => key(1 + (octave * 7 + slot + 1) * 13.85 - 4.2, 8.4, 33, pc, true));
  }
  $("keys").replaceChildren(svg);
}

// ---------------------------------------------------------------- transpose

async function transposeBy(step) {
  const next = Math.max(-11, Math.min(11, state.transpose + step));
  if (next === state.transpose || !state.id) return;
  try {
    const body = await api(`/v1/transcriptions/${state.id}?notes=false&transpose=${next}`);
    state.transpose = next;
    applyResult(body.result);
    if (!$("view-sheet").hidden) renderSheet();
  } catch (error) {
    $("sheet-status").textContent = error.message;
  }
}
$("down").addEventListener("click", () => transposeBy(-1));
$("up").addEventListener("click", () => transposeBy(1));

// ---------------------------------------------------------------- tabs + sheet

function selectTab(name) {
  for (const tab of ["chords", "sheet"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`view-${tab}`).hidden = tab !== name;
  }
  if (name === "sheet") renderSheet();
}
$("tab-chords").addEventListener("click", () => selectTab("chords"));
$("tab-sheet").addEventListener("click", () => selectTab("sheet"));

for (const button of document.querySelectorAll("[data-arrangement]")) {
  button.addEventListener("click", () => {
    state.arrangement = button.dataset.arrangement;
    for (const other of document.querySelectorAll("[data-arrangement]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    applyResult(state.result);
    renderSheet();
  });
}
$("print").addEventListener("click", () => window.print());

let engraver = null;
function loadEngraver() {
  // The notation renderer is large, so it is only fetched when first needed.
  engraver = engraver || new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/static/vendor/opensheetmusicdisplay.min.js";
    script.onload = resolve;
    script.onerror = () => { engraver = null; reject(new Error("Could not load the sheet music renderer.")); };
    document.head.append(script);
  });
  return engraver;
}

async function renderSheet() {
  const key = `${state.id}|${state.arrangement}|${state.transpose}`;
  if (state.sheetKey === key) return;
  state.sheetKey = key;
  const status = $("sheet-status");
  status.textContent = "Engraving the sheet music…";
  try {
    const [xml] = await Promise.all([
      api(`/v1/transcriptions/${state.id}/musicxml?arrangement=${state.arrangement}&transpose=${state.transpose}`),
      loadEngraver(),
    ]);
    if (state.sheetKey !== key) return; // the user moved on while this loaded
    if (!state.osmd) {
      state.osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay($("sheet"), {
        backend: "svg",
        autoResize: true,
        autoBeam: true,
        drawTitle: true,
        drawComposer: true,
        drawPartNames: false,
        drawMeasureNumbers: true,
      });
    }
    await state.osmd.load(xml);
    if (state.sheetKey !== key) return;
    state.osmd.render();
    status.textContent = "";
  } catch (error) {
    if (state.sheetKey === key) {
      state.sheetKey = null;
      status.textContent = error.message || "The sheet music could not be drawn.";
    }
  }
}

// ---------------------------------------------------------------- start over / deep links

$("again").addEventListener("click", () => {
  releaseAudio();
  state.input = null;
  state.id = null;
  state.result = null;
  history.replaceState(null, "", params.has("embed") ? "?embed=1" : location.pathname);
  $("search").value = "";
  $("results").replaceChildren();
  show("start");
});

(async function restore() {
  const id = params.get("t");
  if (!id) return;
  show("working");
  $("working-name").textContent = "your song";
  try {
    const body = await api(`/v1/transcriptions/${encodeURIComponent(id)}?notes=false`);
    if (body.status !== "completed") throw new Error("That transcription is not available.");
    present(body, null);
  } catch (error) {
    history.replaceState(null, "", location.pathname);
    fail(error.message);
  }
})();
