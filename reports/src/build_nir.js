const pptxgen = require("pptxgenjs");
const path = require("path");

const IMG = (f) => path.join(__dirname, "img", f);

// ---- palette (Forest & Moss — matches the Phase 0 deck) ----
const DARK = "16271D", DARK2 = "1F3527";
const FOREST = "2C5F2D", MOSS = "8FB45C", MOSSD = "6E9248";
const PAPER = "F7F6EF", CARD = "FFFFFF", CREAM = "EDEADD";
const INK = "242C26", MUTE = "6F7C71", WHITE = "FFFFFF";
const FAIL = "B0472E", WARN = "C08A2A", WIN = "3E7D44";
const CODE = "Consolas", HEAD = "Georgia", BODY = "Calibri";
const W = 13.333, H = 7.5, M = 0.6;

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.author = "ML / Data Engineering";
pres.title = "SWISSIMAGE RS (NIR) — Delivery Handling Report";

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.18 });
function fit(ow, oh, mw, mh) { const r = Math.min(mw / ow, mh / oh); return { w: ow * r, h: oh * r }; }
function header(s, eyebrow, title) {
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.52, w: 0.12, h: 0.78, fill: { color: MOSS } });
  s.addText(eyebrow.toUpperCase(), { x: M + 0.28, y: 0.5, w: 11, h: 0.26, fontFace: BODY, fontSize: 11, color: MOSSD, bold: true, charSpacing: 3, margin: 0 });
  s.addText(title, { x: M + 0.27, y: 0.8, w: 12.0, h: 0.6, fontFace: HEAD, fontSize: 27, bold: true, color: INK, margin: 0 });
}
function pageNum(s, n) { s.addText(String(n).padStart(2, "0"), { x: W - 1.0, y: H - 0.5, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTE, align: "right" }); }
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: CREAM, width: 1 }, shadow: shadow() });
}
function framedImg(s, file, ow, oh, bx, by, bw, bh, cap) {
  const f = fit(ow, oh, bw, bh - (cap ? 0.32 : 0));
  const ix = bx + (bw - f.w) / 2, iy = by + (bh - (cap ? 0.32 : 0) - f.h) / 2;
  s.addImage({ path: IMG(file), x: ix, y: iy, w: f.w, h: f.h, shadow: shadow() });
  if (cap) s.addText(cap, { x: bx, y: by + bh - 0.3, w: bw, h: 0.3, fontFace: BODY, fontSize: 9.5, italic: true, color: MUTE, align: "center", margin: 0 });
}

/* ============ 1 TITLE ============ */
let s = pres.addSlide(); s.background = { color: DARK };
s.addImage({ path: IMG("RS_cir_triptych.png"), x: 6.9, y: 2.25, w: 6.43, h: 2.16, sizing: { type: "cover", w: 6.43, h: 2.16 } });
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: FOREST } });
s.addText("SESSION REPORT  ·  24 JULY 2026", { x: M, y: 1.25, w: 8, h: 0.35, fontFace: BODY, fontSize: 13, color: MOSS, bold: true, charSpacing: 3 });
s.addText("The Near-Infrared\nData Arrived", { x: M, y: 1.68, w: 8, h: 1.75, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE });
s.addText("Handling the SWISSIMAGE RS delivery: what came, why nothing would open, and what the NIR band actually reveals.", { x: M, y: 4.75, w: 6.0, h: 1.1, fontFace: BODY, fontSize: 15, color: CREAM, lineSpacingMultiple: 1.15 });
s.addShape(pres.shapes.LINE, { x: M, y: 6.1, w: 3.2, h: 0, line: { color: MOSSD, width: 1.5 } });
s.addText([{ text: "Prepared for: ", options: { color: MUTE } }, { text: "Project Lead", options: { color: WHITE, bold: true } }],
  { x: M, y: 6.25, w: 7, h: 0.4, fontFace: BODY, fontSize: 13 });

/* ============ 2 WHAT ARRIVED ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "The delivery", "What swisstopo sent");
const stats = [["25", "tiles", "4-band NIR/R/G/B, 16-bit, 10 cm"], ["6", "dates", "incl. leaf-off 24 Mar 2021"], ["66 GB", "on disk", "flight strips, ~1 GB each"]];
stats.forEach((st, i) => {
  const x = M + i * 4.07;
  card(s, x, 1.65, 3.8, 1.8);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 3.8, h: 0.1, fill: { color: FOREST } });
  s.addText(st[0], { x, y: 1.78, w: 3.8, h: 0.74, fontFace: HEAD, fontSize: 40, bold: true, color: FOREST, align: "center", margin: 0 });
  s.addText(st[1].toUpperCase(), { x, y: 2.56, w: 3.8, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: MOSSD, align: "center", charSpacing: 2, margin: 0 });
  s.addText(st[2], { x: x + 0.2, y: 2.86, w: 3.4, h: 0.5, fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });
});
card(s, M, 3.75, 5.95, 2.55);
s.addText("Not what we asked for — better", { x: M + 0.35, y: 3.95, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: FOREST, margin: 0 });
s.addText([
  { text: "We requested a single 1 km² tile to keep cost down.", options: { color: INK, bullet: true, breakLine: true } },
  { text: "They sent 25 strips across 6 acquisition dates.", options: { color: INK, bullet: true, breakLine: true } },
  { text: "Crucially that includes a leaf-off March flight — the exact phenology needed to separate evergreen from dormant deciduous.", options: { color: INK, bullet: true } },
], { x: M + 0.35, y: 4.4, w: 5.3, h: 1.8, fontFace: BODY, fontSize: 12, lineSpacingMultiple: 1.05, paraSpaceAfter: 5, margin: 0 });
card(s, M + 6.18, 3.75, 5.95, 2.55, DARK);
s.addText("BAND ORDER (CONFIRMED)", { x: M + 6.53, y: 3.95, w: 5.3, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2, margin: 0 });
[["Band 1", "NIR  808–882 nm", MOSS], ["Band 2", "Red", CREAM], ["Band 3", "Green", CREAM], ["Band 4", "Blue", CREAM]].forEach((b, i) => {
  const y = 4.4 + i * 0.40;
  s.addText(b[0], { x: M + 6.53, y, w: 1.3, h: 0.36, fontFace: CODE, fontSize: 12, bold: true, color: b[2], valign: "middle", margin: 0 });
  s.addText(b[1], { x: M + 7.9, y, w: 3.9, h: 0.36, fontFace: BODY, fontSize: 12, color: b[2], valign: "middle", margin: 0 });
});
s.addText("QGIS labels these 'Band 1 (Red)' etc. — that is display-slot naming, not content.", { x: M + 6.53, y: 6.0, w: 5.3, h: 0.28, fontFace: BODY, fontSize: 9.5, italic: true, color: MUTE, margin: 0 });
pageNum(s, 2);

/* ============ 3 TIF + TFW ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "File anatomy", "Why every .tif has a twin .tfw");
card(s, M, 1.65, 5.95, 2.3);
s.addText(".tif — the pixels", { x: M + 0.35, y: 1.82, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: FOREST, margin: 0 });
s.addText("A grid of numbers: 4 bands × ~10 000 × ~20 000 uint16 values. On its own it knows nothing about where on Earth it belongs — it is just an image.", { x: M + 0.35, y: 2.28, w: 5.3, h: 1.5, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
card(s, M + 6.18, 1.65, 5.95, 2.3);
s.addText(".tfw — the world file", { x: M + 6.53, y: 1.82, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: FOREST, margin: 0 });
s.addText("Six plain-text numbers that pin those pixels to the map. Same basename, different extension — that is the entire link. Lose it and the image floats free.", { x: M + 6.53, y: 2.28, w: 5.3, h: 1.5, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
card(s, M, 4.15, 12.13, 2.25, DARK);
s.addText("20240810_0840_12504_0_18.tfw", { x: M + 0.4, y: 4.3, w: 5, h: 0.3, fontFace: CODE, fontSize: 12, color: MOSS, margin: 0 });
const tfw = [["0.100", "pixel size in X — 10 cm"], ["0.00000", "rotation (0 = north-up)"], ["0.00000", "rotation (0 = north-up)"],
["-0.100", "pixel size in Y — negative: rows run north→south"], ["2717648.750", "E of the upper-left pixel centre"], ["1116990.850", "N of the upper-left pixel centre"]];
tfw.forEach((t, i) => {
  const y = 4.68 + i * 0.28;
  s.addText(t[0], { x: M + 0.4, y, w: 1.7, h: 0.26, fontFace: CODE, fontSize: 12, color: WHITE, align: "right", valign: "middle", margin: 0 });
  s.addText("→  " + t[1], { x: M + 2.3, y, w: 9.2, h: 0.26, fontFace: BODY, fontSize: 11.5, color: CREAM, valign: "middle", margin: 0 });
});
s.addText("Together they are an affine transform: pixel (col,row) ⇄ map (E,N). That is all georeferencing is.", { x: M, y: 6.55, w: 12.13, h: 0.35, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, align: "center" });
pageNum(s, 3);

/* ============ 4 THE CRS GAP ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Problem 1", "The tiles knew where — but not on which planet");
card(s, M, 1.7, 12.13, 1.5, DARK);
s.addText([
  { text: "A .tfw gives coordinates but never says which coordinate system they are in.", options: { bold: true, color: WHITE, breakLine: true } },
  { text: "All 25 tiles reported  crs = None.  The numbers were right; the label was missing — so QGIS had no idea they were Swiss LV95 and placed them nowhere. That is what looked like \"the coordinates are off\".", options: { color: CREAM } },
], { x: M + 0.4, y: 1.88, w: 11.3, h: 1.2, fontFace: BODY, fontSize: 13.5, lineSpacingMultiple: 1.15, margin: 0 });
const fixes = [["Symptom", "Tiles invisible or landing in the ocean; rasterio reports crs=None", FAIL],
["Diagnosis", "Georeferencing lives only in the sidecar .tfw; no CRS embedded in the GeoTIFF", WARN],
["Fix", "Stamp EPSG:2056 into each tile's metadata — in-place, metadata-only, no pixel rewrite", WIN]];
fixes.forEach((f, i) => {
  const y = 3.45 + i * 1.05;
  card(s, M, y, 12.13, 0.9);
  s.addShape(pres.shapes.RECTANGLE, { x: M, y, w: 0.12, h: 0.9, fill: { color: f[2] } });
  s.addText(f[0], { x: M + 0.45, y, w: 1.8, h: 0.9, fontFace: HEAD, fontSize: 14, bold: true, color: f[2], valign: "middle", margin: 0 });
  s.addText(f[1], { x: M + 2.4, y, w: 9.5, h: 0.9, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", margin: 0 });
});
s.addText("Done now, not later: georeferencing belongs with the data, so every tool — QGIS, GDAL, the pipeline — gets it right for free. The 33 GB original zip is untouched.", { x: M, y: 6.65, w: 12.13, h: 0.3, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, align: "center" });
pageNum(s, 4);

/* ============ 5 THE inspect.py BUG ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Problem 2", "Why the checking script would not run");
card(s, M, 1.7, 12.13, 1.35, DARK);
s.addText("AttributeError: module 'inspect' has no attribute 'signature'", { x: M + 0.4, y: 1.85, w: 11.3, h: 0.4, fontFace: CODE, fontSize: 14, color: FAIL, margin: 0 });
s.addText("Thrown from deep inside rasterio's imports — nothing to do with the data, and the traceback points nowhere near the real cause.", { x: M + 0.4, y: 2.3, w: 11.3, h: 0.6, fontFace: BODY, fontSize: 12.5, color: CREAM, margin: 0 });
const steps = [["1", "The script was named  inspect.py"], ["2", "Python puts the script's own folder FIRST on the import path"],
["3", "rasterio does  import inspect  → gets the local file, not the standard library"], ["4", "That file was empty, so  inspect.signature  does not exist → rasterio dies"]];
steps.forEach((st, i) => {
  const y = 3.35 + i * 0.72;
  s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.5, h: 0.5, fill: { color: FOREST } });
  s.addText(st[0], { x: M, y, w: 0.5, h: 0.5, fontFace: HEAD, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(st[1], { x: M + 0.75, y, w: 7.6, h: 0.5, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", margin: 0 });
});
card(s, M + 8.6, 3.3, 3.53, 3.0, CREAM);
s.addText("The rule", { x: M + 8.9, y: 3.5, w: 3, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: FOREST, margin: 0 });
s.addText("Never name a file after a standard-library module.", { x: M + 8.9, y: 3.9, w: 2.95, h: 0.8, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
s.addText("Also risky:", { x: M + 8.9, y: 4.75, w: 2.95, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MUTE, margin: 0 });
s.addText("random.py · json.py\nemail.py · types.py\nlogging.py · io.py", { x: M + 8.9, y: 5.05, w: 2.95, h: 1.0, fontFace: CODE, fontSize: 11.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
s.addText("Replaced by  scripts/check_swissimage_rs.py  — reports bands, CRS, dates and which tiles cover the AOI.", { x: M, y: 6.55, w: 12.13, h: 0.35, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, align: "center" });
pageNum(s, 5);

/* ============ 6 WHAT NIR SHOWS ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "The payoff", "What the NIR band actually reveals");
card(s, M, 1.68, 12.13, 3.05);
framedImg(s, "RS_cir_triptych.png", 2392, 805, M + 0.15, 1.8, 11.83, 2.8, "Same 180 m of leaf-off forest, 24 March 2021 — three renderings of the same four bands");
const reads = [["True colour", "Dormant forest is a flat brown mush. This is the RGB-only problem that stalled us.", MUTE],
["Colour-infrared", "Live evergreen foliage flares red against the dark leafless canopy.", FOREST],
["NDVI", "Discrete green dots — individual evergreen crowns — resolved against tan dormant forest.", WIN]];
reads.forEach((r, i) => {
  const x = M + i * 4.07;
  card(s, x, 4.95, 3.8, 1.55);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 4.95, w: 3.8, h: 0.09, fill: { color: r[2] } });
  s.addText(r[0], { x: x + 0.25, y: 5.1, w: 3.3, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: r[2], margin: 0 });
  s.addText(r[1], { x: x + 0.25, y: 5.45, w: 3.35, h: 0.95, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
});
s.addText("This is the discrimination RGB could not provide — exactly the case made in the data request.", { x: M, y: 6.65, w: 12.13, h: 0.35, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, align: "center" });
pageNum(s, 6);

/* ============ 7 COVERAGE GAP ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "The catch", "It does not cover our confirmed palms");
card(s, M, 1.7, 6.0, 4.85);
framedImg(s, "RS_footprint_gap.png", 869, 1160, M + 0.15, 1.82, 5.7, 4.6, null);
s.addText([
  { text: "The sample is centred on the AOI I chose from GBIF point density — not on the palms we personally verified.", options: { color: INK, breakLine: true } },
], { x: M + 6.35, y: 1.85, w: 5.78, h: 0.9, fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.15, margin: 0 });
const gaps = [["Covered", "Phase 0 AOI + 169 GBIF occurrence points", WIN],
["Not covered", "Via Tamporiva — the palms you confirmed in person (~15 km south)", FAIL],
["Not covered", "Usignolo playground — the Street-View-confirmed palm (~12 km west)", FAIL]];
gaps.forEach((g, i) => {
  const y = 2.95 + i * 1.05;
  card(s, M + 6.35, y, 5.78, 0.92);
  s.addShape(pres.shapes.RECTANGLE, { x: M + 6.35, y, w: 0.1, h: 0.92, fill: { color: g[2] } });
  s.addText(g[0].toUpperCase(), { x: M + 6.6, y: y + 0.12, w: 5.4, h: 0.28, fontFace: BODY, fontSize: 10, bold: true, color: g[2], charSpacing: 1.5, margin: 0 });
  s.addText(g[1], { x: M + 6.6, y: y + 0.4, w: 5.4, h: 0.45, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
});
card(s, M + 6.35, 6.1, 5.78, 0.85, CREAM);
s.addText([{ text: "Action: ", options: { bold: true, color: FOREST } },
{ text: "name E 2 719 475 / N 1 095 726 explicitly in the next request.", options: { color: INK } }],
  { x: M + 6.6, y: 6.2, w: 5.4, h: 0.65, fontFace: BODY, fontSize: 12, valign: "middle", margin: 0 });
s.addText("So the delivery validates the mechanism, not a specific palm.", { x: M, y: 6.72, w: 6.0, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, align: "center" });
pageNum(s, 7);

/* ============ 8 REORGANISATION ============ */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Housekeeping", "Codebase reorganised around the new data");
card(s, M, 1.7, 5.9, 4.5, DARK);
s.addText("BEFORE — scattered", { x: M + 0.35, y: 1.9, w: 5.2, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: FAIL, charSpacing: 2, margin: 0 });
s.addText("palmTrack/\n  Sample_SWISSIMAGE_RS/     66 GB loose\n  Sample_SWISSIMAGE_RS.zip\n  check_rs.py               outside repo\n  inspect.py                empty, broken\n  deck/                     outside repo\n  .vscode/  .DS_Store       cruft\n  files/\n    ticino-palm-mapper/     the actual repo",
  { x: M + 0.35, y: 2.3, w: 5.3, h: 3.7, fontFace: CODE, fontSize: 10.5, color: CREAM, lineSpacingMultiple: 1.25, margin: 0 });
card(s, M + 6.23, 1.7, 5.9, 4.5, DARK);
s.addText("AFTER — inside the repo", { x: M + 6.58, y: 1.9, w: 5.2, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2, margin: 0 });
s.addText("ticino-palm-mapper/\n  data/raw/swissimage_rs/\n    lugano_delivery_2026-07/   25 tiles\n    bern_sample_2019/\n  scripts/\n    00–05 pipeline\n    check_swissimage_rs.py\n  reports/\n    *.pptx  +  src/ build tooling\n  src/  configs/  docs/",
  { x: M + 6.58, y: 2.3, w: 5.3, h: 3.7, fontFace: CODE, fontSize: 10.5, color: CREAM, lineSpacingMultiple: 1.25, margin: 0 });
card(s, M, 6.35, 12.13, 0.75, CREAM);
s.addText([{ text: "Committed & pushed. ", options: { bold: true, color: FOREST } },
{ text: "The 66 GB of imagery and node_modules are gitignored — the commit is 24 MB of code, reports and figures.", options: { color: INK } }],
  { x: M + 0.35, y: 6.45, w: 11.5, h: 0.55, fontFace: BODY, fontSize: 12, valign: "middle", margin: 0 });
pageNum(s, 8);

/* ============ 9 STATUS / NEXT ============ */
s = pres.addSlide(); s.background = { color: DARK };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: FOREST } });
s.addText("STATUS  ·  NEXT", { x: M, y: 0.85, w: 10, h: 0.35, fontFace: BODY, fontSize: 13, color: MOSS, bold: true, charSpacing: 3 });
s.addText("NIR is in hand and it works.", { x: M, y: 1.25, w: 11.5, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE });
card(s, M, 2.45, 5.9, 2.35, DARK2);
s.addText("DONE THIS SESSION", { x: M + 0.35, y: 2.62, w: 5, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2, margin: 0 });
s.addText([
  { text: "25 RS tiles inventoried, CRS repaired", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Broken checker diagnosed and replaced", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "NDVI resolves individual evergreen crowns", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Repo reorganised, committed, pushed", options: { color: CREAM, bullet: true } },
], { x: M + 0.35, y: 2.98, w: 5.3, h: 1.7, fontFace: BODY, fontSize: 12, lineSpacingMultiple: 1.1, margin: 0 });
card(s, M + 6.23, 2.45, 5.9, 2.35, DARK2);
s.addText("NEXT", { x: M + 6.58, y: 2.62, w: 5, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2, margin: 0 });
s.addText([
  { text: "Retile RS strips onto the 10 cm grid", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Extend stacks to [R,G,B,NIR,NDVI,CHM]", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Request coverage over a confirmed palm", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "ML: Dataset, model, training (project lead)", options: { color: MOSS, bullet: true } },
], { x: M + 6.58, y: 2.98, w: 5.3, h: 1.7, fontFace: BODY, fontSize: 12, lineSpacingMultiple: 1.1, margin: 0 });
s.addShape(pres.shapes.LINE, { x: M, y: 5.15, w: 12.1, h: 0, line: { color: MOSSD, width: 1 } });
s.addText("The blocker was never the imagery — it was knowing what the pixels meant.", { x: M, y: 5.45, w: 11.5, h: 0.7, fontFace: HEAD, fontSize: 16, italic: true, color: CREAM });
s.addText([{ text: "Repo:  ", options: { color: MUTE } }, { text: "github.com/aidenfairman12/ticino-palm-mapper", options: { color: MOSS, bold: true } }],
  { x: M, y: 6.3, w: 12, h: 0.4, fontFace: BODY, fontSize: 12.5 });

pres.writeFile({ fileName: path.join(__dirname, "..", "SWISSIMAGE_RS_NIR_Report.pptx") })
  .then((f) => console.log("WROTE", f));
