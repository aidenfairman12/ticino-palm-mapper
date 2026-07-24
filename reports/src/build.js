const pptxgen = require("pptxgenjs");
const path = require("path");

const IMG = (f) => path.join(__dirname, "img", f);

// ---- palette (Forest & Moss, topic-fit) ----
const DARK = "16271D", DARK2 = "1F3527";
const FOREST = "2C5F2D", FORESTD = "234A24";
const MOSS = "8FB45C", MOSSD = "6E9248";
const PAPER = "F7F6EF", CARD = "FFFFFF", CREAM = "EDEADD";
const INK = "242C26", MUTE = "6F7C71", WHITE = "FFFFFF";
const FAIL = "B0472E", PARTIAL = "C08A2A", BLOCK = "8A8F8A", WIN = "3E7D44";

const HEAD = "Georgia", BODY = "Calibri";
const W = 13.333, H = 7.5, M = 0.6;

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.author = "ML / Data Engineering";
pres.title = "Ticino Palm Mapper — Phase 0 Session Report";

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.18 });
function fit(ow, oh, mw, mh) {
  const r = Math.min(mw / ow, mh / oh);
  return { w: ow * r, h: oh * r };
}
function header(s, eyebrow, title, titleColor) {
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.52, w: 0.12, h: 0.78, fill: { color: MOSS } });
  s.addText(eyebrow.toUpperCase(), { x: M + 0.28, y: 0.5, w: 11, h: 0.3, fontFace: BODY, fontSize: 11, color: MOSSD, bold: true, charSpacing: 3, margin: 0 });
  s.addText(title, { x: M + 0.27, y: 0.78, w: 12.0, h: 0.62, fontFace: HEAD, fontSize: 27, bold: true, color: titleColor || INK, margin: 0 });
}
function pageNum(s, n) {
  s.addText(String(n).padStart(2, "0"), { x: W - 1.0, y: H - 0.5, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTE, align: "right" });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: CREAM, width: 1 }, shadow: shadow() });
}
function framedImg(s, file, ow, oh, bx, by, bw, bh, cap) {
  const f = fit(ow, oh, bw, bh - (cap ? 0.32 : 0));
  const ix = bx + (bw - f.w) / 2, iy = by + (bh - (cap ? 0.32 : 0) - f.h) / 2;
  s.addImage({ path: IMG(file), x: ix, y: iy, w: f.w, h: f.h, shadow: shadow() });
  if (cap) s.addText(cap, { x: bx, y: by + bh - 0.3, w: bw, h: 0.3, fontFace: BODY, fontSize: 9.5, italic: true, color: MUTE, align: "center", margin: 0 });
}

/* ================= 1 — TITLE ================= */
let s = pres.addSlide(); s.background = { color: DARK };
s.addImage({ path: IMG("REF_ground_photo.jpg"), x: 8.45, y: 0, w: 4.88, h: H, sizing: { type: "cover", w: 4.88, h: H } });
s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y: 0, w: 0.06, h: H, fill: { color: MOSS } });
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: FOREST } });
s.addText("SESSION REPORT  ·  21 JUNE 2026", { x: M, y: 1.5, w: 7.4, h: 0.35, fontFace: BODY, fontSize: 13, color: MOSS, bold: true, charSpacing: 3 });
s.addText("Mapping an Invasive Palm\nfrom the Sky", { x: M, y: 1.95, w: 7.5, h: 1.9, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, lineSpacingMultiple: 1.0 });
s.addText("Ticino Palm Mapper — Phase 0: data pipeline built, and a hard look at how we will actually label the palms.", { x: M, y: 3.95, w: 7.2, h: 1.0, fontFace: BODY, fontSize: 15, color: CREAM, lineSpacingMultiple: 1.15 });
s.addShape(pres.shapes.LINE, { x: M, y: 5.35, w: 3.2, h: 0, line: { color: MOSSD, width: 1.5 } });
s.addText([
  { text: "Prepared for: ", options: { color: MUTE } }, { text: "Project Lead", options: { color: WHITE, bold: true, breakLine: true } },
  { text: "From: ", options: { color: MUTE } }, { text: "ML / Data Engineering", options: { color: WHITE, bold: true } },
], { x: M, y: 5.5, w: 7, h: 0.8, fontFace: BODY, fontSize: 13 });

/* ================= 2 — EXEC SUMMARY ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "The one-slide version", "What happened this session");
const stats = [
  ["6", "scripts", "Phase 0 pipeline, end-to-end on live data"],
  ["7,118", "GBIF records", "free palm occurrences evaluated as labels"],
  ["3", "imagery layers", "RGB + LiDAR height + 2018/21/24 temporal"],
];
stats.forEach((st, i) => {
  const x = M + i * 4.07;
  card(s, x, 1.7, 3.8, 1.85);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w: 3.8, h: 0.1, fill: { color: FOREST } });
  s.addText(st[0], { x, y: 1.85, w: 3.8, h: 0.85, fontFace: HEAD, fontSize: 44, bold: true, color: FOREST, align: "center", margin: 0 });
  s.addText(st[1].toUpperCase(), { x, y: 2.68, w: 3.8, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: MOSSD, align: "center", charSpacing: 2, margin: 0 });
  s.addText(st[2], { x: x + 0.25, y: 2.98, w: 3.3, h: 0.5, fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });
});
card(s, M, 3.95, 12.13, 2.85, DARK);
s.addText("THE BINDING CONSTRAINT IS LABELING", { x: M + 0.4, y: 4.2, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: MOSS, charSpacing: 2 });
s.addText([
  { text: "Not imagery, not compute — labels.", options: { bold: true, color: WHITE } },
  { text: " The plan assumed we could eyeball palms in the orthophotos and hand-label from them. We stress-tested that and it does not hold: at 10 cm RGB, individual crowns are not reliably separable by eye — even against a confirmed, ground-truthed palm.", options: { color: CREAM } },
], { x: M + 0.4, y: 4.6, w: 11.3, h: 1.0, fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.15, margin: 0 });
s.addText([
  { text: "The way forward: ", options: { bold: true, color: MOSS } },
  { text: "fuse weak signals (RGB texture + LiDAR height + temporal persistence) in a trained model, and build the test set from ground truth — not from eyeballing.", options: { color: CREAM } },
], { x: M + 0.4, y: 5.75, w: 11.3, h: 0.9, fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.15, margin: 0 });
pageNum(s, 2);

/* ================= 3 — WHAT WE SHIPPED ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Deliverable", "An end-to-end Phase 0 data layer");
const steps = [
  ["00", "Fetch SWISSIMAGE 10 cm RGB tiles (swisstopo STAC COGs, memory-safe)"],
  ["01", "Fetch occurrence points from the GBIF API, reprojected to EPSG:2056"],
  ["02", "Align points to the tile grid (point GeoJSON + raster masks)"],
  ["03", "Add co-registered LiDAR height → [R,G,B,CHM] feature stacks"],
  ["04", "Fetch the 2018 / 2021 / 2024 time series (aligned per tile)"],
  ["05", "HTML labeling-assist sheet: crop + GBIF + Street View links"],
];
steps.forEach((st, i) => {
  const y = 1.75 + i * 0.78;
  s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.56, h: 0.56, fill: { color: i >= 3 ? MOSSD : FOREST } });
  s.addText(st[0], { x: M, y, w: 0.56, h: 0.56, fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(st[1], { x: M + 0.78, y: y - 0.02, w: 6.0, h: 0.6, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", margin: 0 });
});
card(s, 7.7, 1.7, 5.03, 4.55);
framedImg(s, "contact_sheet.png", 1070, 1337, 7.85, 1.82, 4.73, 4.0, "Pipeline output: aligned RGB tiles + occurrence points");
s.addText([
  { text: "Shipped to a private GitHub repo  ", options: { color: INK } },
  { text: "(github.com/aidenfairman12/ticino-palm-mapper)", options: { color: MOSSD, italic: true } },
  { text: ".  Data is gitignored; ML modules left as stubs for the ML owner.", options: { color: MUTE } },
], { x: M, y: 6.55, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 11.5, align: "left", margin: 0 });
pageNum(s, 3);

/* ================= 4 — WHY LABELING ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Framing", "Why labeling is the whole ballgame");
const why = [
  ["A model is only as honest as its test set", "Report precision/recall on a small, exhaustively-labelled hold-out set — or the headline map is unverifiable."],
  ["Occurrence points are presence-only", "They say 'a palm was seen near here', not which crown, and not 'every palm in this tile'. Absence of a point ≠ absence of a palm."],
  ["So the labels must be earned", "Cheap label sources exist; trustworthy ones do not come for free. This session was about finding out how trustworthy each source really is."],
];
why.forEach((w3, i) => {
  const y = 1.85 + i * 1.5;
  card(s, M, y, 12.13, 1.32);
  s.addShape(pres.shapes.RECTANGLE, { x: M, y, w: 0.12, h: 1.32, fill: { color: FOREST } });
  s.addText(w3[0], { x: M + 0.45, y: y + 0.16, w: 11.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: FOREST, margin: 0 });
  s.addText(w3[1], { x: M + 0.45, y: y + 0.6, w: 11.4, h: 0.6, fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
});
pageNum(s, 4);

/* ================= 5 — EVOLUTION TIMELINE ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "The evolution", "Five attempts at labeling, one lesson");
const stages = [
  ["1", "Info Flora\nmanual export", "No coordinates", FAIL],
  ["2", "GBIF API\noccurrence points", "Too imprecise", PARTIAL],
  ["3", "Eyeball the\northophoto", "Can't see them", FAIL],
  ["4", "Ground photo +\nStreet View", "Confirms, can't\nlocalise", PARTIAL],
  ["5", "Multi-signal\nfusion", "Promising", WIN],
];
const cw = 2.18, gap = 0.30, x0 = M + 0.1;
stages.forEach((st, i) => {
  const x = x0 + i * (cw + gap);
  card(s, x, 2.15, cw, 3.4);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.15, w: cw, h: 0.12, fill: { color: st[3] } });
  s.addShape(pres.shapes.OVAL, { x: x + cw / 2 - 0.4, y: 2.5, w: 0.8, h: 0.8, fill: { color: st[3] } });
  s.addText(st[0], { x: x + cw / 2 - 0.4, y: 2.5, w: 0.8, h: 0.8, fontFace: HEAD, fontSize: 26, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(st[1], { x: x + 0.1, y: 3.5, w: cw - 0.2, h: 1.0, fontFace: HEAD, fontSize: 14.5, bold: true, color: INK, align: "center", valign: "top", margin: 0, lineSpacingMultiple: 1.0 });
  s.addText(st[2], { x: x + 0.1, y: 4.55, w: cw - 0.2, h: 0.9, fontFace: BODY, fontSize: 12, italic: true, color: st[3], align: "center", valign: "top", margin: 0 });
  if (i < stages.length - 1) s.addText("→", { x: x + cw - 0.02, y: 2.15, w: gap, h: 3.4, fontFace: BODY, fontSize: 22, color: MUTE, align: "center", valign: "middle", margin: 0 });
});
s.addText("Each source was abandoned for a concrete reason — the next slide is the scorecard.", { x: M, y: 5.85, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: MUTE });
pageNum(s, 5);

/* ================= 6 — SCORECARD TABLE ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Methods scorecard", "What we tried, and why each fell short");
const hc = (t) => ({ text: t, options: { fill: { color: FOREST }, color: WHITE, bold: true, fontSize: 11.5, fontFace: BODY, valign: "middle", align: "left" } });
const vc = (t, col) => ({ text: t, options: { color: WHITE, fill: { color: col }, bold: true, fontSize: 10.5, align: "center", valign: "middle" } });
const rows = [
  [hc("Method"), hc("The promise"), hc("Why it fell short / its limit"), hc("Verdict")],
  ["Info Flora 'AtlasWS' export", "Official Swiss flora data", "It's a per-square species checklist — no coordinates at all", vc("UNUSABLE", FAIL)],
  ["GBIF occurrence points", "7,118 free CH points, with coords", "55% at 18 m (=180 px); a '1 m' point landed on a rooftop", vc("WEAK ANCHOR", PARTIAL)],
  ["Eyeball the RGB orthophoto", "Free, needs no labels", "Crowns not separable from native/ornamental at 10 cm; forest occlusion", vc("UNRELIABLE", FAIL)],
  ["iNaturalist photo match", "Confirms the species", "Photos are close-ups — no scene context to match to the ortho", vc("CAN'T LOCALISE", FAIL)],
  ["Street View cross-reference", "Oblique, eye-level ID", "Confirms a palm, but RGB ortho still ambiguous; leaf-off branches mimic palms", vc("GT ONLY", PARTIAL)],
  ["NIR (SWISSIMAGE RS)", "Spectral separability", "Not freely available — request / paid only", vc("BLOCKED", BLOCK)],
  ["LiDAR canopy height (CHM)", "Free; separates veg vs ground", "A 5 m palm looks like a 5 m shrub — height alone can't ID", vc("FEATURE", PARTIAL)],
  ["Multi-signal fusion", "Combines the weak signals", "Needs a trained model + a ground-truthed test set", vc("RECOMMENDED", WIN)],
];
const body = rows.map((r, ri) => r.map((c) => {
  if (typeof c !== "string") return c;
  return { text: c, options: { fontSize: 10.5, fontFace: BODY, color: INK, valign: "middle", fill: { color: ri % 2 ? PAPER : CARD } } };
}));
s.addTable(body, { x: M, y: 1.75, w: 12.13, colW: [2.7, 2.9, 4.63, 1.9], rowH: 0.52, border: { type: "solid", pt: 0.5, color: CREAM }, align: "left", valign: "middle", autoPage: false });
pageNum(s, 6);

/* ================= 7 — PROOF: POINTS MISS ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Evidence · 1", "Occurrence points do not sit on palms");
card(s, M, 1.75, 5.95, 4.05);
framedImg(s, "REF_confirmed_from_above.png", 2057, 1052, M + 0.15, 1.9, 5.65, 3.75, "A GBIF point tagged 1 m uncertainty — lands on a rooftop");
card(s, M + 6.18, 1.75, 5.95, 4.05);
framedImg(s, "chips_native.png", 1657, 1662, M + 6.33, 1.9, 5.65, 3.75, "Most points carry 18 m uncertainty; markers fall on generic canopy");
card(s, M, 6.0, 12.13, 0.95, DARK);
s.addText([
  { text: "Takeaway:  ", options: { bold: true, color: MOSS } },
  { text: "'coordinate uncertainty' reflects the phone's GPS accuracy, not the plant's true position. Treat all occurrence points as tile-level weak positives — never as crown labels.", options: { color: CREAM } },
], { x: M + 0.4, y: 6.18, w: 11.3, h: 0.6, fontFace: BODY, fontSize: 13.5, valign: "middle", margin: 0 });
pageNum(s, 7);

/* ================= 8 — PROOF: CONFIRMED YET AMBIGUOUS ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Evidence · 2", "A confirmed palm — still can't be picked out from above");
card(s, M, 1.75, 5.4, 4.05);
framedImg(s, "REF_ground_photo.jpg", 2048, 1365, M + 0.15, 1.9, 5.1, 3.75, "Street View → iNaturalist confirms Trachycarpus at a known spot");
card(s, M + 5.63, 1.75, 6.5, 4.05);
framedImg(s, "REF_brissago_zooms.png", 2139, 1462, M + 5.78, 1.9, 6.2, 3.75, "Yet in 10 cm RGB — here a botanical garden full of palms — the radial crown isn't separable by eye");
card(s, M, 6.0, 12.13, 0.95, DARK);
s.addText([
  { text: "Takeaway:  ", options: { bold: true, color: MOSS } },
  { text: "leaf-off imagery doesn't rescue it either — a bare deciduous tree's branches radiate like palm fronds and produce false positives. The eye (and RGB alone) is not enough.", options: { color: CREAM } },
], { x: M + 0.4, y: 6.18, w: 11.3, h: 0.6, fontFace: BODY, fontSize: 13.5, valign: "middle", margin: 0 });
pageNum(s, 8);

/* ================= 9 — WHAT WORKS ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Evidence · 3", "The signal is in the combination");
card(s, M, 1.7, 12.13, 2.5);
framedImg(s, "REF_USER_PALM.png", 3346, 883, M + 0.15, 1.82, 11.83, 2.26, "User-pinpointed palm — RGB 2018/2021/2024 + LiDAR height. Each panel alone is weak; together they agree.");
card(s, M, 4.35, 5.5, 2.55);
framedImg(s, "coreg_check.png", 1347, 833, M + 0.15, 4.47, 5.2, 2.3, "CHM co-registered to the 10 cm RGB grid");
s.addText([
  { text: "Three weak signals that agree:", options: { bold: true, color: FOREST, breakLine: true, fontSize: 15 } },
  { text: "Evergreen & persistent", options: { bold: true, color: INK, bullet: { indent: 18 }, breakLine: true } },
  { text: "present across all three vintages (deciduous & construction are not).", options: { color: MUTE, breakLine: true } },
  { text: "~3 m height in the CHM", options: { bold: true, color: INK, bullet: { indent: 18 }, breakLine: true } },
  { text: "a real object, not lawn or shadow.", options: { color: MUTE, breakLine: true } },
  { text: "Plausibly radial crown", options: { bold: true, color: INK, bullet: { indent: 18 }, breakLine: true } },
  { text: "consistent with a fan palm.", options: { color: MUTE } },
], { x: M + 5.85, y: 4.5, w: 6.25, h: 2.4, fontFace: BODY, fontSize: 12.5, lineSpacingMultiple: 1.05, paraSpaceAfter: 4, margin: 0 });
pageNum(s, 9);

/* ================= 10 — DATA AVAILABILITY ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "What the data allows", "Free signals vs the one that isn't");
const layers = [
  ["RGB · SWISSIMAGE 10 cm", "FREE", WIN, "Resolves crowns by size (~20–30 px). The base layer."],
  ["LiDAR height (CHM)", "FREE", WIN, "DSM−DTM. Masks ground, gives height. A feature, not an ID."],
  ["Temporal 2018/21/24", "FREE", WIN, "Evergreen persistence vs change. Aligns per tile for free."],
  ["NIR · SWISSIMAGE RS", "REQUEST / PAID", BLOCK, "Likely the single best discriminator — but not freely served."],
];
layers.forEach((l, i) => {
  const x = M + (i % 2) * 6.18, y = 1.85 + Math.floor(i / 2) * 1.78;
  card(s, x, y, 5.95, 1.55);
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 5.95, h: 0.1, fill: { color: l[2] } });
  s.addText(l[0], { x: x + 0.3, y: y + 0.22, w: 4.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText(l[1], { x: x + 3.7, y: y + 0.24, w: 2.05, h: 0.34, fontFace: BODY, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle", fill: { color: l[2] }, margin: 0 });
  s.addText(l[3], { x: x + 0.3, y: y + 0.72, w: 5.35, h: 0.7, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
});
card(s, M, 5.5, 12.13, 1.4, CREAM);
s.addText([
  { text: "Design note — ", options: { bold: true, color: FOREST } },
  { text: "at 10 cm, two adjacent palms merge into one blob (we saw this on a pin that was actually two trees). ", options: { color: INK } },
  { text: "Favour density / counting over exact instance separation in Phase 1.", options: { bold: true, color: INK } },
], { x: M + 0.4, y: 5.7, w: 11.3, h: 1.0, fontFace: BODY, fontSize: 14, valign: "middle", lineSpacingMultiple: 1.15, margin: 0 });
pageNum(s, 10);

/* ================= 11 — RECOMMENDATIONS ================= */
s = pres.addSlide(); s.background = { color: PAPER };
header(s, "Recommendations", "Five moves for the next phase");
const recs = [
  ["Make height + temporal core inputs", "Move CHM and the multi-vintage stack into Phase 1 — not the 'optional v2' the plan assumed."],
  ["Request SWISSIMAGE RS (NIR) now", "Long lead time, and likely the strongest single discriminator for evergreen palms."],
  ["Ground-truth the test set", "Build it with the Street View labeling tool (script 05). Occurrence points stay as weak positives only."],
  ["Target density first", "Counting / density map as Phase 1; precise individual detection as the stretch goal."],
  ["Budget for labeling, not compute", "Compute is ample. Trustworthy labels are the real cost — plan the effort explicitly."],
];
recs.forEach((r, i) => {
  const y = 1.75 + i * 1.02;
  s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.62, h: 0.62, fill: { color: FOREST } });
  s.addText(String(i + 1), { x: M, y, w: 0.62, h: 0.62, fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(r[0], { x: M + 0.85, y: y - 0.03, w: 11.2, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText(r[1], { x: M + 0.85, y: y + 0.38, w: 11.2, h: 0.5, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
});
pageNum(s, 11);

/* ================= 12 — STATUS & NEXT (close) ================= */
s = pres.addSlide(); s.background = { color: DARK };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: FOREST } });
s.addText("STATUS  ·  NEXT", { x: M, y: 0.85, w: 10, h: 0.35, fontFace: BODY, fontSize: 13, color: MOSS, bold: true, charSpacing: 3 });
s.addText("Phase 0 is done. The data layer is ready.", { x: M, y: 1.25, w: 11.5, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE });
card(s, M, 2.5, 5.9, 2.0, DARK2);
s.addText("DELIVERED", { x: M + 0.35, y: 2.7, w: 5, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2 });
s.addText([
  { text: "6-script pipeline: RGB + LiDAR + temporal + labels", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Labeling feasibility characterised end-to-end", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Pushed to a private GitHub repo", options: { color: CREAM, bullet: true } },
], { x: M + 0.35, y: 3.05, w: 5.3, h: 1.3, fontFace: BODY, fontSize: 12.5, lineSpacingMultiple: 1.1, margin: 0 });
card(s, M + 6.23, 2.5, 5.9, 2.0, DARK2);
s.addText("NEXT (ML OWNER)", { x: M + 6.58, y: 2.7, w: 5, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MOSS, charSpacing: 2 });
s.addText([
  { text: "Dataset over the [R,G,B,CHM] feature stacks", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Baseline density / counting detector", options: { color: CREAM, bullet: true, breakLine: true } },
  { text: "Hand-label seed + ground-truthed hold-out set", options: { color: CREAM, bullet: true } },
], { x: M + 6.58, y: 3.05, w: 5.3, h: 1.3, fontFace: BODY, fontSize: 12.5, lineSpacingMultiple: 1.1, margin: 0 });
s.addShape(pres.shapes.LINE, { x: M, y: 5.0, w: 12.1, h: 0, line: { color: MOSSD, width: 1 } });
s.addText([
  { text: "Repo:  ", options: { color: MUTE } },
  { text: "github.com/aidenfairman12/ticino-palm-mapper", options: { color: MOSS, bold: true } },
  { text: "  (private)", options: { color: MUTE } },
], { x: M, y: 5.25, w: 12, h: 0.4, fontFace: BODY, fontSize: 13 });
s.addText("Honest feasibility work is itself a result: we now know exactly where the hard part is.", { x: M, y: 5.95, w: 11.5, h: 0.7, fontFace: HEAD, fontSize: 15, italic: true, color: CREAM });

pres.writeFile({ fileName: path.join(__dirname, "Ticino_Palm_Mapper_Phase0_Report.pptx") })
  .then((f) => console.log("WROTE", f));
