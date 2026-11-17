"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.has("embed")) document.documentElement.classList.add("embed");

const SHOW_KEYS = ["notes", "chords", "lyrics", "key", "tempo"];

const state = {
  id: null,
  source: null,
  doc: null, // the editable score document (see chordlistener/document.py)
  view: null, // how the server says the chords, key and tempo read right now
  baseline: "", // the document as last saved, to know whether anything changed
  canSave: false,
  edited: false,
  editing: false,
  undo: [],
  redo: [],
  sel: null, // selected note or rest on the sheet
  hits: [], // clickable things on the sheet
  transpose: 0,
  arrangement: "piano",
  show: loadShow(),
  objectUrl: null,
  input: null, // what was analysed, so it can be re-run with other settings
  tempo: "auto",
  capo: false, // showing the chords a guitarist would play (capo or dropped tuning)
  lyrics: null, // what the lyrics lookup found for this song
  links: {},
  maxUploadMb: 1000,
  osmd: null,
  sheetStale: true,
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
  if (response.status === 204) return null;
  const isJson = (response.headers.get("content-type") || "").includes("json");
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok && response.status !== 202) {
    const message = (body && body.error && body.error.message) || "Something went wrong. Please try again.";
    throw new Error(message);
  }
  return body;
}

function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private browsing: carry on without remembering */ }
  return null;
}

function loadShow() {
  const saved = store("cl-show") || {};
  const out = {};
  for (const key of SHOW_KEYS) out[key] = saved[key] !== false;
  return out;
}

// The token that proves this browser created a transcription and may save changes to it.
function tokenHeader() {
  const tokens = store("cl-edit-tokens") || {};
  return tokens[state.id] ? { "X-Edit-Token": tokens[state.id] } : {};
}

function rememberToken(id, token) {
  const tokens = store("cl-edit-tokens") || {};
  tokens[id] = token;
  const ids = Object.keys(tokens);
  for (const old of ids.slice(0, Math.max(0, ids.length - 60))) delete tokens[old];
  store("cl-edit-tokens", tokens);
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = el("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function fileStem() {
  const title = (state.doc && state.doc.title) || "transcription";
  return title.replace(/[^\p{L}\p{N} _-]+/gu, "_").trim().slice(0, 80) || "transcription";
}

const scripts = {};
function loadScript(src) {
  // Large libraries are only fetched when first needed.
  scripts[src] = scripts[src] || new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => { delete scripts[src]; reject(new Error("Could not load part of the page. Please try again.")); };
    document.head.append(script);
  });
  return scripts[src];
}
const loadEngraver = () => loadScript("/static/vendor/opensheetmusicdisplay.min.js");

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

/** POST a form, reporting upload progress (fetch cannot do that yet). */
function send(path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => { if (event.lengthComputable) onProgress(event.loaded / event.total); });
    request.addEventListener("load", () => {
      const body = request.response;
      if (request.status >= 400 && request.status !== 422) {
        reject(new Error((body && body.error && body.error.message) || "Something went wrong. Please try again."));
      } else if (!body) {
        reject(new Error("The server sent an answer that could not be read."));
      } else {
        resolve(body);
      }
    });
    request.addEventListener("error", () => reject(new Error("Could not reach the server. Check your connection and try again.")));
    request.send(form);
  });
}

api("/v1/health").then((health) => { state.maxUploadMb = health.max_upload_mb || state.maxUploadMb; }).catch(() => {});

async function submit() {
  const input = state.input;
  if (input.file && input.file.size > state.maxUploadMb * 1024 * 1024) {
    fail(`That file is ${Math.round(input.file.size / 1048576)} MB. The largest upload accepted here is ${state.maxUploadMb} MB.`);
    return;
  }
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
  const big = Boolean(input.file) && input.file.size > 8 * 1024 * 1024;
  $("working-step").textContent = big ? "Uploading" : "Listening to";
  $("working-progress").hidden = !big;
  $("working-progress").value = 0;
  show("working");
  try {
    let body = await send("/v1/transcriptions", form, (done) => {
      $("working-progress").value = Math.round(done * 100);
      if (done >= 1) {
        $("working-step").textContent = "Listening to";
        $("working-progress").hidden = true;
      }
    });
    if (body.edit_token) rememberToken(body.id, body.edit_token);
    while (body.status === "processing") {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      body = await api(`/v1/transcriptions/${body.id}`);
    }
    if (body.status !== "completed") throw new Error((body.error && body.error.message) || "The analysis failed.");
    await present(body, input.audioUrl);
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
  if (isDirty() && !confirm("Re-analysing at a different tempo starts a fresh transcription. Your unsaved corrections will be lost. Continue?")) return;
  $("audio").pause();
  state.tempo = next;
  submit();
}
$("half").addEventListener("click", () => retime(-1));
$("double").addEventListener("click", () => retime(1));

function convertedAudio() {
  const token = (store("cl-edit-tokens") || {})[state.id];
  return state.links.audio && token ? `${state.links.audio}?token=${encodeURIComponent(token)}` : "";
}

function releaseAudio() {
  const audio = $("audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
}

// ---------------------------------------------------------------- result

async function present(body, audioUrl) {
  state.id = body.id;
  state.source = body.source;
  state.links = body.links || {};
  state.transpose = 0;
  state.capo = false;
  state.lyrics = null;
  state.undo = [];
  state.redo = [];
  state.sel = null;
  setEditing(false, true);
  await loadScore();

  const source = body.source;
  $("artwork").hidden = !source.artwork_url;
  if (source.artwork_url) $("artwork").src = source.artwork_url;
  $("clip-note").hidden = source.type !== "catalog";

  const tuning = body.result && body.result.tuning;
  const offPitch = Boolean(tuning) && Math.abs(tuning.cents) >= 10;
  $("chip-tuning").hidden = !offPitch;
  if (offPitch) {
    $("fact-tuning").textContent = `A = ${Math.round(tuning.reference_hz)} Hz`;
    $("chip-tuning").title = `This recording is ${Math.abs(tuning.cents)} cents ${tuning.cents < 0 ? "flat" : "sharp"} of standard pitch. Chords and notes are shown at the nearest standard pitch.`;
  }

  // Without the original file to hand (a shared link, or a video the browser
  // cannot play), fall back to the MP3 the server made from an uploaded video.
  const url = audioUrl || source.preview_url || convertedAudio() || "";
  $("player").classList.toggle("silent", !url);
  if (url && $("audio").getAttribute("src") !== url) $("audio").src = url;
  $("half").hidden = $("double").hidden = !state.input;
  $("half").disabled = state.tempo === "half";
  $("double").disabled = state.tempo === "double";
  setPlaying(false);
  $("seek").value = 0;

  $("sheet").replaceChildren();
  state.osmd = null;
  state.sheetStale = true;
  selectTab("chords");
  draw();
  show("result");
  loadLyrics();
  history.replaceState(null, "", `?${params.has("embed") ? "embed=1&" : ""}t=${encodeURIComponent(body.id)}`);
  window.scrollTo({ top: 0 });
}

async function loadScore(original) {
  const score = await api(`/v1/transcriptions/${state.id}/score${original ? "?original=true" : ""}`, { headers: tokenHeader() });
  state.doc = score.document;
  state.view = score.view;
  state.canSave = score.can_save;
  state.edited = score.edited && !original;
  state.baseline = JSON.stringify(state.doc);
}

/** Bring everything on the page in line with state.doc / state.view. */
function draw() {
  const { doc, view } = state;
  $("song-title").textContent = doc.title || "Untitled";
  $("song-artist").textContent = doc.artist || "";
  $("fact-key").textContent = view.key.name;
  $("fact-tempo").textContent = `${Math.round(view.tempo_bpm)} bpm`;
  $("fact-time").textContent = view.time_signature;
  $("semitones").textContent = (state.transpose > 0 ? "+" : "") + state.transpose;
  $("seek").max = Math.max(view.duration, (state.source && state.source.audio_seconds) || 0);
  applyShow();
  renderChart();
  drawGuitarTip();
  drawLyricsPanel();
  drawEditBar();
  state.lastBeat = -1;
  state.lastChord = -2;
  sync();
}

// A guitarist rarely plays the sounding chords of a song in B flat minor:
// offer the capo position (or dropped tuning) that turns them into easy shapes.
function drawGuitarTip() {
  const tip = state.view.guitar;
  const applied = Boolean(tip) && state.capo && state.transpose === tip.transpose;
  const how = tip && (tip.capo ? `a capo on fret ${tip.capo}` : "the guitar tuned down a half step");
  $("guitar-tip").hidden = !tip || state.editing;
  if (tip) {
    $("tip-text").textContent = applied
      ? `Showing the chords as played with ${how}. They sound in ${state.doc ? keyName(0) : "the original key"}.`
      : `Guitar: these chords are easier with ${how} — ${tip.chords.join(", ")}.`;
    $("tip-apply").textContent = applied ? "Show the sounding chords" : "Show those chords";
  }
  const label = $("capo-label");
  label.hidden = !applied;
  if (applied) label.textContent = tip.capo ? `Capo ${tip.capo}` : "Tuned ½ step down";
}

function keyName(shift) {
  const key = state.doc.key;
  const names = key.minor ? MINOR_KEYS : MAJOR_KEYS;
  return `${names[(key.tonic + shift + 12) % 12]} ${key.minor ? "minor" : "major"}`;
}

$("tip-apply").addEventListener("click", () => {
  const tip = state.view.guitar;
  if (!tip) return;
  const applied = state.capo && state.transpose === tip.transpose;
  state.capo = !applied;
  state.transpose = applied ? 0 : tip.transpose;
  refresh();
});

function capoNumber() {
  const tip = state.view && state.view.guitar;
  return tip && state.capo && state.transpose === tip.transpose ? tip.capo : 0;
}

// ---------------------------------------------------------------- lyrics

async function loadLyrics() {
  const id = state.id;
  try {
    const found = await api(`/v1/transcriptions/${id}/lyrics`, { headers: tokenHeader() });
    if (state.id === id) { state.lyrics = found; drawLyricsPanel(); }
  } catch { /* lyrics are a bonus */ }
}

/** Lyrics that could not be lined up with the bars are still worth showing, as plain text. */
function drawLyricsPanel() {
  const found = state.lyrics;
  const placed = state.doc && Object.keys(state.doc.lyrics).length > 0;
  const panel = $("lyrics-panel");
  panel.hidden = !found || !found.found || placed;
  if (panel.hidden) return;
  const excerpt = state.source && state.source.type === "catalog";
  $("lyrics-note").textContent = excerpt
    ? "These are the words of the whole song. What you are hearing is a 30-second excerpt, so they could not be placed under the bars; upload the full track and they will be."
    : "These could not be lined up with the bars automatically. Use Edit, then Find lyrics or Paste lyrics, to place them.";
  const text = $("lyrics-text");
  text.replaceChildren();
  let last = null;
  for (const line of found.lines) {
    if (last !== null && line.time !== null && line.time - last > 9) text.append(el("p", "gap"));
    text.append(el("p", "", line.text));
    last = line.time;
  }
  $("lyrics-source").textContent = found.source === "lrclib" ? "Lyrics from LRCLIB (lrclib.net)." : `Lyrics from ${found.source}.`;
}

function applyShow() {
  for (const button of document.querySelectorAll("[data-show]")) {
    button.setAttribute("aria-pressed", String(state.show[button.dataset.show]));
  }
  $("chip-key").hidden = !state.show.key;
  $("chip-tempo").hidden = !state.show.tempo;
  $("result").classList.toggle("hide-chords", !state.show.chords);
  $("result").classList.toggle("hide-lyrics", !state.show.lyrics);
  document.querySelector(".segmented").hidden = !state.show.notes; // a chord sheet has no piano/lead choice
}

for (const button of document.querySelectorAll("[data-show]")) {
  button.addEventListener("click", () => {
    const key = button.dataset.show;
    state.show[key] = !state.show[key];
    store("cl-show", state.show);
    applyShow();
    state.sel = null;
    state.sheetStale = true;
    if (!$("view-sheet").hidden) refresh();
  });
}

function renderChart() {
  const { doc, view } = state;
  const perBar = doc.beats_per_bar;
  const starts = new Map(); // global beat index -> chord symbol as shown
  for (const chord of view.chords) starts.set((chord.bar - 1) * perBar + (chord.beat - 1), chord.chord);

  const chart = $("chart");
  chart.replaceChildren();
  state.beatEls = [];
  state.barEls = [];
  let carried = "";
  for (let bar = 0; bar < view.bars; bar++) {
    const cell = el("div", "bar");
    cell.append(el("span", "n", String(bar + 1)));
    const beats = el("div", "beats");
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
      if (state.editing) span.addEventListener("click", () => editChord(index, span));
      beats.append(span);
      state.beatEls.push(span);
    }
    if (changes > 2) cell.classList.add("dense");
    cell.append(beats);

    const words = doc.lyrics[String(bar)] || "";
    if (state.editing) {
      const input = el("input", "lyric-input");
      input.type = "text";
      input.value = words;
      input.maxLength = 200;
      input.placeholder = "lyrics";
      input.setAttribute("aria-label", `Lyrics for bar ${bar + 1}`);
      input.addEventListener("change", () => setLyric(bar, input.value));
      cell.append(input);
    } else {
      cell.append(el("div", "lyric", words));
      cell.addEventListener("click", () => seekTo(view.beats[bar * perBar]));
    }
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
audio.addEventListener("timeupdate", () => sync());
audio.addEventListener("error", () => {
  const current = audio.getAttribute("src");
  if (!current) return;
  const fallback = convertedAudio();
  if (fallback && current !== fallback) audio.src = fallback; // e.g. an AVI: play the converted MP3 instead
  else $("player").classList.add("silent");
});
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
  const view = state.view;
  if (!view) return;
  const time = typeof at === "number" ? at : audio.currentTime || 0;
  if (document.activeElement !== $("seek")) $("seek").value = time;
  $("clock").textContent = `${clock(time)} / ${clock(parseFloat($("seek").max))}`;

  const beat = Math.min(lastAtOrBefore(view.beats, time), state.beatEls.length - 1);
  if (beat !== state.lastBeat) {
    const perBar = state.doc.beats_per_bar;
    const before = state.beatEls[state.lastBeat];
    if (before) before.classList.remove("on");
    const oldBar = state.barEls[Math.floor(state.lastBeat / perBar)];
    if (oldBar) oldBar.classList.remove("active");
    const now = state.beatEls[beat];
    if (now) now.classList.add("on");
    const bar = state.barEls[Math.floor(beat / perBar)];
    if (bar) {
      bar.classList.add("active");
      if (!audio.paused && !$("view-chords").hidden && !state.editing) bar.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    state.lastBeat = beat;
  }

  const chords = view.chords;
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

// ---------------------------------------------------------------- rendering through the server

function renderRequest(extra) {
  return api("/v1/score/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document: state.doc, arrangement: state.arrangement, transpose: state.transpose, capo: capoNumber(), show: state.show, ...extra }),
  });
}

let refreshSeq = 0;
let refreshTimer = 0;

/** Ask the server how the current document reads, and redraw. Returns false if the document was refused. */
async function refresh() {
  clearTimeout(refreshTimer);
  const seq = ++refreshSeq;
  const wantSheet = !$("view-sheet").hidden;
  if (wantSheet) $("sheet-status").textContent = "Updating…";
  let out;
  try {
    out = await renderRequest({ musicxml: wantSheet });
  } catch (error) {
    if (seq === refreshSeq) $("sheet-status").textContent = error.message;
    return false;
  }
  if (seq !== refreshSeq) return true; // something newer is on its way
  state.doc = out.document;
  state.view = out.view;
  draw();
  if (wantSheet) await drawSheet(out.musicxml);
  else state.sheetStale = true;
  return true;
}

function refreshSoon() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refresh, 160);
}

// ---------------------------------------------------------------- transpose

function transposeBy(step) {
  if (state.editing) return;
  const next = Math.max(-11, Math.min(11, state.transpose + step));
  if (next === state.transpose || !state.id) return;
  state.transpose = next;
  state.capo = false; // a hand-picked transposition is not a capo position
  refresh();
}
$("down").addEventListener("click", () => transposeBy(-1));
$("up").addEventListener("click", () => transposeBy(1));

// ---------------------------------------------------------------- tabs + sheet

function selectTab(name) {
  for (const tab of ["chords", "sheet"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`view-${tab}`).hidden = tab !== name;
  }
  if (name === "sheet" && state.sheetStale && state.doc) refresh();
  drawInspector();
}
$("tab-chords").addEventListener("click", () => selectTab("chords"));
$("tab-sheet").addEventListener("click", () => selectTab("sheet"));

for (const button of document.querySelectorAll("[data-arrangement]")) {
  button.addEventListener("click", () => {
    state.arrangement = button.dataset.arrangement;
    for (const other of document.querySelectorAll("[data-arrangement]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    state.sel = null;
    refresh();
  });
}

function engraverOptions(extra) {
  return {
    backend: "svg",
    autoResize: false,
    autoBeam: true,
    drawTitle: true,
    drawSubtitle: true,
    drawComposer: true,
    drawPartNames: false,
    drawMeasureNumbers: true,
    ...extra,
  };
}

function newEngraver(container, extra) {
  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, engraverOptions(extra));
  try {
    // "C6" is how musicians write it; the renderer's default is "Cmaj6".
    osmd.EngravingRules.ChordSymbolLabelTexts.setValue(opensheetmusicdisplay.ChordSymbolEnum.majorsixth, "6");
  } catch { /* cosmetic only */ }
  return osmd;
}

let drawSeq = 0;
async function drawSheet(xml) {
  const seq = ++drawSeq;
  const status = $("sheet-status");
  try {
    await loadEngraver();
    if (!state.osmd) state.osmd = newEngraver($("sheet"));
    await state.osmd.load(xml);
    if (seq !== drawSeq) return;
    state.osmd.zoom = window.innerWidth < 720 ? 0.7 : 0.85;
    state.osmd.render();
    state.sheetStale = false;
    status.textContent = sheetHint();
    bindSheet();
  } catch (error) {
    if (seq === drawSeq) status.textContent = error.message || "The sheet music could not be drawn.";
  }
}

function sheetHint() {
  if (!state.editing) return "";
  return state.show.notes ? "Click a note or a rest to change it." : "Turn on Notes to edit the notes.";
}

let resizeTimer = 0;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.osmd && !$("view-sheet").hidden && !state.sheetStale) {
      state.osmd.render();
      bindSheet();
    }
  }, 250);
});

// ---------------------------------------------------------------- editing: the basics

function isDirty() {
  return Boolean(state.doc) && JSON.stringify(state.doc) !== state.baseline;
}

/** Change the document. Every change goes through here so that it can be undone. */
function edit(change, redraw = "all") {
  const before = JSON.stringify(state.doc);
  change(state.doc);
  if (JSON.stringify(state.doc) === before) return;
  state.undo.push(before);
  if (state.undo.length > 60) state.undo.shift();
  state.redo = [];
  state.sheetStale = true;
  drawEditBar();
  drawInspector();
  if (redraw === "all") refreshSoon();
}

async function step(from, to) {
  if (!from.length) return;
  to.push(JSON.stringify(state.doc));
  state.doc = JSON.parse(from.pop());
  state.sel = null;
  await refresh();
}
$("undo").addEventListener("click", () => step(state.undo, state.redo));
$("redo").addEventListener("click", () => step(state.redo, state.undo));

function setEditing(on, quiet) {
  state.editing = on;
  state.sel = null;
  $("result").classList.toggle("editing", on);
  $("editbar").hidden = !on;
  $("edit").hidden = on;
  $("down").disabled = $("up").disabled = on;
  if (quiet) return;
  if (on) { state.transpose = 0; state.capo = false; } // corrections are made at the original pitch
  state.sheetStale = true;
  refresh();
}

$("edit").addEventListener("click", () => setEditing(true));
$("done").addEventListener("click", async () => {
  if (isDirty() && state.canSave && confirm("Save your changes before you finish?")) await save();
  setEditing(false);
});

function drawEditBar() {
  if (!state.doc) return;
  const dirty = isDirty();
  const status = $("edit-status");
  status.classList.toggle("warn", dirty && state.canSave);
  if (!state.canSave) status.textContent = "You can make changes and download them. Only the person who created this transcription can save them here.";
  else if (dirty) status.textContent = "Unsaved changes";
  else status.textContent = state.edited ? "All changes saved" : "";
  $("undo").disabled = !state.undo.length;
  $("redo").disabled = !state.redo.length;
  $("save").disabled = !dirty || !state.canSave;
  $("revert").disabled = !state.edited && !state.undo.length;
  const fields = { "ed-title": state.doc.title, "ed-artist": state.doc.artist, "ed-bpm": Math.round(state.doc.bpm), "ed-key": `${state.doc.key.tonic}:${state.doc.key.minor ? 1 : 0}` };
  for (const [id, value] of Object.entries(fields)) {
    if (document.activeElement !== $(id)) $(id).value = value;
  }
}

async function save() {
  try {
    const out = await api(`/v1/transcriptions/${state.id}/score`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...tokenHeader() },
      body: JSON.stringify(state.doc),
    });
    state.doc = out.document;
    state.baseline = JSON.stringify(state.doc);
    state.edited = true;
    drawEditBar();
    return true;
  } catch (error) {
    $("edit-status").textContent = error.message;
    $("edit-status").classList.add("warn");
    return false;
  }
}
$("save").addEventListener("click", save);

$("revert").addEventListener("click", async () => {
  if (!confirm("Throw away all corrections and go back to the automatic transcription?")) return;
  try {
    if (state.canSave && state.edited) await api(`/v1/transcriptions/${state.id}/score`, { method: "DELETE", headers: tokenHeader() });
    await loadScore(!state.canSave);
    state.undo = [];
    state.redo = [];
    state.sel = null;
    state.sheetStale = true;
    await refresh();
  } catch (error) {
    $("edit-status").textContent = error.message;
  }
});

window.addEventListener("beforeunload", (event) => {
  if (state.canSave && isDirty()) event.preventDefault();
});

// Song details
const MAJOR_KEYS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const MINOR_KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"];
for (const [names, minor] of [[MAJOR_KEYS, 0], [MINOR_KEYS, 1]]) {
  names.forEach((name, tonic) => {
    const option = el("option", "", `${name} ${minor ? "minor" : "major"}`);
    option.value = `${tonic}:${minor}`;
    $("ed-key").append(option);
  });
}
$("ed-title").addEventListener("change", () => edit((doc) => { doc.title = $("ed-title").value.trim(); }));
$("ed-artist").addEventListener("change", () => edit((doc) => { doc.artist = $("ed-artist").value.trim(); }));
$("ed-key").addEventListener("change", () => edit((doc) => {
  const [tonic, minor] = $("ed-key").value.split(":").map(Number);
  doc.key = { tonic, minor: Boolean(minor) };
}));
$("ed-bpm").addEventListener("change", () => edit((doc) => {
  const bpm = Math.round(parseFloat($("ed-bpm").value));
  if (bpm >= 20 && bpm <= 400) doc.bpm = bpm;
}));

// ---------------------------------------------------------------- editing: chords and lyrics

function editChord(beat, span) {
  if (span.querySelector("input")) return;
  const existing = state.doc.chords.find((c) => c[0] === beat);
  const input = el("input");
  input.type = "text";
  input.maxLength = 24;
  input.value = existing ? existing[1] : "";
  input.placeholder = "chord";
  input.setAttribute("aria-label", "Chord");
  span.classList.remove("held");
  span.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = (commit) => {
    if (finished) return;
    finished = true;
    const text = input.value.trim();
    if (commit && text !== (existing ? existing[1] : "")) {
      edit((doc) => {
        doc.chords = doc.chords.filter((c) => c[0] !== beat);
        if (text) doc.chords.push([beat, text]);
        doc.chords.sort((a, b) => a[0] - b[0]);
      });
    } else {
      renderChart();
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

function setLyric(bar, text) {
  // The chart already shows what was typed, so there is nothing to redraw
  // (and redrawing would take the cursor away from the next box).
  edit((doc) => {
    const clean = text.replace(/\s+/g, " ").trim();
    if (clean) doc.lyrics[String(bar)] = clean;
    else delete doc.lyrics[String(bar)];
  }, "none");
  state.view.has_lyrics = Object.keys(state.doc.lyrics).length > 0;
}

$("find-lyrics").addEventListener("click", async () => {
  const button = $("find-lyrics");
  const status = $("edit-status");
  if (!state.doc.title) { status.textContent = "Fill in the title (and artist) first, then try again."; return; }
  button.disabled = true;
  button.textContent = "Searching…";
  try {
    const form = new FormData();
    form.append("title", state.doc.title);
    form.append("artist", state.doc.artist);
    const found = await api(`/v1/transcriptions/${state.id}/lyrics`, { method: "POST", body: form, headers: tokenHeader() });
    state.lyrics = found;
    if (!found.found) {
      status.textContent = `No lyrics found for "${state.doc.title}". Check the title and artist, or paste them in.`;
    } else if (found.bars && Object.keys(found.bars).length) {
      edit((doc) => { doc.lyrics = found.bars; });
      status.textContent = "Lyrics added under the bars.";
    } else {
      // Found, but without timings that fit this recording: let the person place them.
      $("ly-text").value = found.lines.map((line) => line.text).join("\n");
      $("ly-start").max = state.view.bars;
      $("lyrics-dialog").showModal();
    }
    drawLyricsPanel();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Find lyrics";
  }
});

$("paste-lyrics").addEventListener("click", () => {
  $("ly-start").max = state.view.bars;
  $("lyrics-dialog").showModal();
});
$("ly-apply").addEventListener("click", () => {
  const lines = $("ly-text").value.split("\n").map((line) => line.trim()).filter(Boolean);
  const span = parseInt($("ly-span").value, 10) || 1;
  const first = Math.max(1, parseInt($("ly-start").value, 10) || 1) - 1;
  edit((doc) => {
    lines.forEach((line, row) => {
      // Share a line's words evenly between the bars it covers.
      const words = line.split(/\s+/);
      for (let part = 0; part < span; part++) {
        const bar = first + row * span + part;
        if (bar >= state.view.bars) return;
        const chunk = words.slice(Math.round((part * words.length) / span), Math.round(((part + 1) * words.length) / span)).join(" ");
        if (chunk) doc.lyrics[String(bar)] = chunk.slice(0, 200);
        else delete doc.lyrics[String(bar)];
      }
    });
  });
  $("lyrics-dialog").close();
  $("ly-text").value = "";
});

// ---------------------------------------------------------------- editing: notes on the sheet

const staffEvents = (staff) => state.doc.arrangements[state.arrangement][staff] || [];
const totalSixteenths = () => (state.doc.beats.length - 1) * 4;

function eventAt(staff, position) {
  const events = staffEvents(staff);
  for (let i = 0; i < events.length; i++) {
    if (events[i][0] <= position && position < events[i][0] + events[i][1]) return i;
    if (events[i][0] > position) break;
  }
  return -1;
}

/** Make every note and rest on the sheet clickable. */
function bindSheet() {
  state.hits = [];
  if (!state.osmd || !state.editing || !state.show.notes) { drawInspector(); return; }
  const rows = state.osmd.GraphicSheet.MeasureList;
  for (const row of rows) {
    row.forEach((measure, staff) => {
      if (!measure) return;
      for (const entry of measure.staffEntries) {
        for (const voice of entry.graphicalVoiceEntries) {
          for (const note of voice.notes) {
            const source = note.sourceNote;
            const element = note.getSVGGElement && note.getSVGGElement();
            if (!element || element.dataset.bound) continue; // the notes of a chord share one element
            const position = Math.round(source.getAbsoluteTimestamp().RealValue * 16);
            const rest = source.isRest();
            const length = Math.max(1, Math.round(source.Length.RealValue * 16));
            element.dataset.bound = "1";
            element.classList.add("cl-hit");
            element.addEventListener("click", () => pick(staff, position, rest, length));
            state.hits.push({ element, staff, position, rest });
          }
        }
      }
    });
  }
  highlight();
  drawInspector();
}

function pick(staff, position, rest, length) {
  const index = eventAt(staff, position);
  if (index >= 0) {
    const event = staffEvents(staff)[index];
    state.sel = { staff, start: event[0], pitch: event[2].length - 1 };
  } else if (rest) {
    state.sel = { staff, start: position, rest: true, length };
  } else {
    state.sel = null;
  }
  highlight();
  drawInspector();
}

function selectedEvent() {
  const sel = state.sel;
  if (!sel || sel.rest) return null;
  const events = staffEvents(sel.staff);
  const index = events.findIndex((e) => e[0] === sel.start);
  return index < 0 ? null : { events, index, event: events[index] };
}

function highlight() {
  const sel = state.sel;
  const found = selectedEvent();
  for (const hit of state.hits) {
    let on = false;
    if (sel && hit.staff === sel.staff) {
      if (sel.rest) on = hit.rest && hit.position === sel.start;
      else if (found) on = !hit.rest && hit.position >= found.event[0] && hit.position < found.event[0] + found.event[1];
    }
    hit.element.classList.toggle("cl-sel", on);
  }
}

const SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
function noteName(pitch) {
  const key = state.doc.key;
  const major = key.minor ? (key.tonic + 3) % 12 : key.tonic;
  const names = [5, 10, 3, 8, 1].includes(major) ? FLATS : SHARPS;
  return names[pitch % 12] + (Math.floor(pitch / 12) - 1);
}

const LENGTHS = { 1: "sixteenth", 2: "eighth", 3: "dotted eighth", 4: "quarter", 6: "dotted quarter", 8: "half", 12: "dotted half", 16: "whole" };
const lengthName = (n) => LENGTHS[n] || `${n} sixteenths`;

function whereText(position) {
  const bar = state.doc.beats_per_bar * 4;
  const inside = position % bar;
  const beat = Math.floor(inside / 4) + 1;
  const part = ["", " and a quarter", " and a half", " and three quarters"][inside % 4];
  return `Bar ${Math.floor(position / bar) + 1}, beat ${beat}${part}`;
}

function iconButton(label, text, action, disabled) {
  const button = el("button", "icon", text);
  button.type = "button";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.disabled = Boolean(disabled);
  button.addEventListener("click", action);
  return button;
}

function drawInspector() {
  const box = $("inspector");
  const sel = state.sel;
  const visible = state.editing && !$("view-sheet").hidden && state.show.notes && sel;
  box.hidden = !visible;
  if (!visible) return;
  box.replaceChildren();
  const staffName = state.arrangement === "piano" ? (sel.staff === 0 ? "right hand" : "left hand") : "melody";

  if (sel.rest) {
    box.append(el("span", "where", `${whereText(sel.start)} · ${staffName} · rest`));
    const add = el("button", "btn small primary", "Add a note here");
    add.type = "button";
    add.addEventListener("click", addNoteAtRest);
    box.append(add);
    return;
  }
  const found = selectedEvent();
  if (!found) { state.sel = null; box.hidden = true; return; }
  const [start, length, pitches] = found.event;
  const next = found.events[found.index + 1];
  const previous = found.events[found.index - 1];
  const room = (next ? next[0] : totalSixteenths()) - start;
  const floor = previous ? previous[0] + previous[1] : 0;
  box.append(el("span", "where", `${whereText(start)} · ${staffName}`));

  const notes = el("span", "group");
  notes.append(el("span", "label", pitches.length > 1 ? "Notes" : "Note"));
  pitches.forEach((pitch, i) => {
    const chip = el("span", "pitch" + (i === sel.pitch ? " current" : ""));
    const name = el("button", "name", noteName(pitch));
    name.type = "button";
    name.title = "Select this note";
    name.addEventListener("click", () => { sel.pitch = i; drawInspector(); });
    chip.append(
      name,
      iconButton("Higher", "▲", () => { sel.pitch = i; nudge(1); }, pitch >= 108),
      iconButton("Lower", "▼", () => { sel.pitch = i; nudge(-1); }, pitch <= 21),
      iconButton("Remove this note", "✕", () => { sel.pitch = i; removePitch(); }),
    );
    notes.append(chip);
  });
  const more = el("button", "btn small", "+ Note");
  more.type = "button";
  more.title = "Add another note to this chord";
  more.disabled = pitches.length >= 8;
  more.addEventListener("click", addPitch);
  notes.append(more);
  box.append(notes);

  const size = el("span", "group");
  size.append(
    el("span", "label", "Length"),
    iconButton("Shorter", "−", () => resize(-1), length <= 1),
    el("span", "", lengthName(length)),
    iconButton("Longer", "+", () => resize(1), length >= room),
  );
  box.append(size);

  const move = el("span", "group");
  move.append(
    el("span", "label", "Move"),
    iconButton("Earlier", "◀", () => shift(-1), start - 1 < floor),
    iconButton("Later", "▶", () => shift(1), start + 1 + length > (next ? next[0] : totalSixteenths())),
  );
  box.append(move);
}

function onSelected(change) {
  edit((doc) => {
    const found = selectedEvent();
    if (found) change(found, doc);
  });
  highlight();
}

function nudge(amount) {
  onSelected(({ event }) => {
    const sel = state.sel;
    const target = event[2][sel.pitch] + amount;
    if (target < 21 || target > 108 || event[2].includes(target)) return;
    event[2][sel.pitch] = target;
    event[2].sort((a, b) => a - b);
    sel.pitch = event[2].indexOf(target);
  });
}

function removePitch() {
  onSelected(({ events, index, event }) => {
    if (event[2].length > 1) {
      event[2].splice(state.sel.pitch, 1);
      state.sel.pitch = Math.min(state.sel.pitch, event[2].length - 1);
    } else {
      events.splice(index, 1);
      state.sel = null;
    }
  });
}

function addPitch() {
  onSelected(({ event }) => {
    let target = event[2][event[2].length - 1] + 3;
    while (event[2].includes(target)) target++;
    if (target > 108) return;
    event[2].push(target);
    state.sel.pitch = event[2].length - 1;
  });
}

function resize(amount) {
  onSelected(({ events, index, event }) => {
    const next = events[index + 1];
    const room = (next ? next[0] : totalSixteenths()) - event[0];
    event[1] = Math.max(1, Math.min(room, event[1] + amount));
  });
}

function shift(amount) {
  onSelected(({ events, index, event }) => {
    const previous = events[index - 1];
    const next = events[index + 1];
    const start = event[0] + amount;
    if (start < (previous ? previous[0] + previous[1] : 0)) return;
    if (start + event[1] > (next ? next[0] : totalSixteenths())) return;
    event[0] = start;
    state.sel.start = start;
  });
}

function addNoteAtRest() {
  const sel = state.sel;
  if (!sel || !sel.rest) return;
  edit((doc) => {
    const events = doc.arrangements[state.arrangement][sel.staff];
    const after = events.findIndex((e) => e[0] > sel.start);
    const next = after < 0 ? null : events[after];
    const before = (after < 0 ? events[events.length - 1] : events[after - 1]) || null;
    const room = (next ? next[0] : totalSixteenths()) - sel.start;
    if (room < 1 || eventAt(sel.staff, sel.start) >= 0) return;
    // Start from the pitch of the neighbouring note: it is usually close.
    const neighbour = before || next;
    const fallback = state.arrangement === "piano" && sel.staff === 1 ? 48 : 67;
    const pitch = neighbour ? neighbour[2][neighbour[2].length - 1] : fallback;
    events.splice(after < 0 ? events.length : after, 0, [sel.start, Math.min(sel.length, room), [pitch]]);
    state.sel = { staff: sel.staff, start: sel.start, pitch: 0 };
  });
  highlight();
}

function stepSelection(direction) {
  const found = selectedEvent();
  if (!found) return;
  const target = found.events[found.index + direction];
  if (!target) return;
  state.sel = { staff: state.sel.staff, start: target[0], pitch: target[2].length - 1 };
  highlight();
  drawInspector();
}

document.addEventListener("keydown", (event) => {
  if (!state.editing || $("result").hidden) return;
  if (document.querySelector("dialog[open]")) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
  const command = event.ctrlKey || event.metaKey;
  if (command && event.key.toLowerCase() === "s") { event.preventDefault(); if (state.canSave) save(); return; }
  if (typing) return;
  if (command && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? step(state.redo, state.undo) : step(state.undo, state.redo); return; }
  if (command && event.key.toLowerCase() === "y") { event.preventDefault(); step(state.redo, state.undo); return; }
  if (!state.sel || state.sel.rest || $("view-sheet").hidden || command) return;
  const actions = {
    ArrowUp: () => nudge(event.shiftKey ? 12 : 1),
    ArrowDown: () => nudge(event.shiftKey ? -12 : -1),
    ArrowLeft: () => stepSelection(-1),
    ArrowRight: () => stepSelection(1),
    Delete: removePitch,
    Backspace: removePitch,
    "[": () => resize(-1),
    "]": () => resize(1),
    Escape: () => { state.sel = null; highlight(); drawInspector(); },
  };
  if (actions[event.key]) { event.preventDefault(); actions[event.key](); }
});

// ---------------------------------------------------------------- download

function downloadChoices() {
  const show = {};
  for (const key of SHOW_KEYS) show[key] = $(`dl-${key}`).checked && !$(`dl-${key}`).disabled;
  const arrangement = document.querySelector('input[name="dl-arrangement"]:checked').value;
  return { show, arrangement };
}

function syncDownloadDialog() {
  const { show } = downloadChoices();
  $("dl-layout").disabled = !show.notes;
  $("dl-pdf").disabled = !(show.notes || show.chords || show.lyrics);
}
for (const key of SHOW_KEYS) $(`dl-${key}`).addEventListener("change", syncDownloadDialog);

$("download").addEventListener("click", () => {
  const hasLyrics = Object.keys(state.doc.lyrics).length > 0;
  for (const key of SHOW_KEYS) $(`dl-${key}`).checked = state.show[key];
  $("dl-lyrics").disabled = !hasLyrics;
  $("dl-lyrics").checked = hasLyrics && state.show.lyrics;
  $("dl-lyrics-hint").textContent = hasLyrics ? "" : "(none yet — add them with Edit)";
  document.querySelector(`input[name="dl-arrangement"][value="${state.arrangement}"]`).checked = true;
  $("dl-midi").href = `/v1/transcriptions/${state.id}/midi?transpose=${state.transpose}`;
  $("dl-json").href = `/v1/transcriptions/${state.id}/score`;
  const mp3 = convertedAudio();
  $("dl-mp3").hidden = $("dl-mp3-dot").hidden = !mp3;
  if (mp3) $("dl-mp3").href = mp3;
  $("dl-error").hidden = true;
  syncDownloadDialog();
  $("download-dialog").showModal();
});

async function withBusy(button, label, task) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  $("dl-error").hidden = true;
  try {
    await task();
  } catch (error) {
    $("dl-error").textContent = error.message || "The download could not be prepared.";
    $("dl-error").hidden = false;
  } finally {
    button.textContent = original;
    button.disabled = false;
    syncDownloadDialog();
  }
}

$("dl-xml").addEventListener("click", () => withBusy($("dl-xml"), "preparing…", async () => {
  const out = await renderRequest({ ...downloadChoices(), musicxml: true });
  saveBlob(new Blob([out.musicxml], { type: "application/vnd.recordare.musicxml+xml" }), `${fileStem()}.musicxml`);
}));

$("dl-chart").addEventListener("click", () => withBusy($("dl-chart"), "preparing…", async () => {
  const out = await renderRequest({ ...downloadChoices(), musicxml: false, chart: true });
  saveBlob(new Blob([out.chart], { type: "text/plain;charset=utf-8" }), `${fileStem()}.chords.txt`);
}));

$("dl-pdf").addEventListener("click", () => withBusy($("dl-pdf"), "Preparing PDF…", async () => {
  const out = await renderRequest({ ...downloadChoices(), musicxml: true });
  const pdf = await buildPdf(out.musicxml);
  saveBlob(pdf.output("blob"), `${fileStem()}.pdf`);
}));

// The PDF's built-in fonts only cover Western European text. Everyday
// typographic characters are swapped for plain ones; anything else (Greek,
// Cyrillic, Chinese…) switches to embedding each page as an image.
const PLAIN = { "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " " };

async function buildPdf(xml) {
  await Promise.all([loadEngraver(), loadScript("/static/vendor/jspdf.umd.min.js")]);
  await loadScript("/static/vendor/svg2pdf.umd.min.js");
  xml = xml.replace(/[‘’“”–—… ]/g, (c) => PLAIN[c]);
  const asImages = /[^\u0000-ÿ]/.test(xml);

  const stage = $("pdf-stage");
  stage.replaceChildren();
  try {
    const osmd = newEngraver(stage, { pageFormat: "A4_P", pageBackgroundColor: "#FFFFFF" });
    await osmd.load(xml);
    osmd.zoom = 0.72; // about four bars to a line on A4
    osmd.render();
    const pages = [...stage.querySelectorAll("svg")];
    if (!pages.length) throw new Error("There is nothing to put in the PDF.");
    // The PDF converter misplaces text that relies on preserved leading spaces
    // (the "= 120" of the tempo mark), so turn those spaces into an offset.
    for (const text of stage.querySelectorAll("text")) {
      if (!text.hasAttribute("xml:space")) continue;
      const lead = text.textContent.length - text.textContent.trimStart().length;
      text.removeAttribute("xml:space");
      text.textContent = text.textContent.trim();
      text.setAttribute("x", parseFloat(text.getAttribute("x")) + lead * 4);
      const size = /^([\d.]+)pt$/.exec(text.getAttribute("font-size") || "");
      if (size) text.setAttribute("font-size", `${(parseFloat(size[1]) * 4) / 3}px`);
    }
    const pdf = new jspdf.jsPDF({ unit: "mm", format: "a4", orientation: "portrait", compress: true });
    const [width, height] = [210, 297];
    for (let i = 0; i < pages.length; i++) {
      if (i) pdf.addPage();
      if (asImages) pdf.addImage(await rasterise(pages[i]), "PNG", 0, 0, width, height, undefined, "FAST");
      else await pdf.svg(pages[i], { x: 0, y: 0, width, height });
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(8);
      pdf.setTextColor(140);
      pdf.text(`${i + 1} / ${pages.length}`, width / 2, height - 7, { align: "center" });
    }
    return pdf;
  } finally {
    stage.replaceChildren();
  }
}

function rasterise(svg) {
  return new Promise((resolve, reject) => {
    const copy = svg.cloneNode(true);
    copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 1654; // A4 at 200 dpi
      canvas.height = 2339;
      const context = canvas.getContext("2d");
      context.fillStyle = "#fff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/png"));
    };
    image.onerror = () => reject(new Error("The PDF page could not be drawn."));
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(copy));
  });
}

// ---------------------------------------------------------------- start over / deep links

$("again").addEventListener("click", () => {
  if (state.canSave && isDirty() && !confirm("You have unsaved corrections. Leave without saving?")) return;
  releaseAudio();
  state.input = null;
  state.id = null;
  state.doc = null;
  state.view = null;
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
    await present(body, null);
  } catch (error) {
    history.replaceState(null, "", location.pathname);
    fail(error.message);
  }
})();
