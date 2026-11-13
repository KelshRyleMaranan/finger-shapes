import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// How easily the tracker accepts a hand (0 to 1). Lower = catches more hands
// but may rarely see a "hand" that isn't one. The default is 0.5.
const TRACKING_SENSITIVITY = 0.3;

// If a hand disappears for a moment, keep its last position for this many
// frames (about 0.2 seconds) so the shape doesn't flicker.
const HOLD_FRAMES = 6;

// The fingers we can detect. tip / joint / base are MediaPipe point numbers.
// A finger counts as "out" when its tip is far from the base compared with
// its middle joint. "on" and "off" are two different limits, so a finger
// that is half bent doesn't flicker on and off.
const FINGERS = [
  { name: "thumb", tip: 4, joint: 3, base: 17, on: 1.1, off: 1.0 },
  { name: "index", tip: 8, joint: 6, base: 0, on: 1.3, off: 1.15 },
  { name: "middle", tip: 12, joint: 10, base: 0, on: 1.3, off: 1.15 },
  { name: "ring", tip: 16, joint: 14, base: 0, on: 1.3, off: 1.15 },
  { name: "pinky", tip: 20, joint: 18, base: 0, on: 1.3, off: 1.15 },
];

// Which fingers can get a line. Remove a name to stop that finger
// from ever counting, for example remove "ring" and "pinky" for fewer lines.
const SHAPE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];

// true  = a closed finger is not drawn at all (no dots, no bones).
// false = the whole hand skeleton is always drawn.
const HIDE_CLOSED_FINGERS = true;

// How thick the lines are, in screen pixels.
// The outline traces the whole shape. The lines inside it are thinner.
const OUTLINE_WIDTH = 4;
const INNER_LINE_WIDTH = 1.5;

// true  = a faint line texture over the shape.
// false = no texture, the shape is only marked by its lines.
const SHOW_SCANLINES = true;

// Where two sections of the shape overlap:
// false = the filter drops out there and the plain picture shows through.
// true  = the filter is drawn once over the whole shape, overlaps included.
const OVERLAP_KEEPS_FILTER = false;

// The hand skeleton. The palm is always drawn. Each finger has its own
// bones and joints (same order as FINGERS), so a closed finger can be hidden.
const PALM_BONES = [[0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
const PALM_JOINTS = [0, 1, 5, 9, 13, 17];
const FINGER_PARTS = [
  { bones: [[1, 2], [2, 3], [3, 4]], joints: [2, 3, 4] },
  { bones: [[5, 6], [6, 7], [7, 8]], joints: [6, 7, 8] },
  { bones: [[9, 10], [10, 11], [11, 12]], joints: [10, 11, 12] },
  { bones: [[13, 14], [14, 15], [15, 16]], joints: [14, 15, 16] },
  { bones: [[17, 18], [18, 19], [19, 20]], joints: [18, 19, 20] },
];

// Camcorder yellow. The person can change it with the color control.
const DEFAULT_COLOR = "#ffd21f";

// Filters that can be applied inside the shape. When the filter control is
// on, one of these is picked at random and changes every few seconds.
const FILTERS = [
  { id: "pixelate", name: "PIXELATE" },
  { id: "glitch", name: "GLITCH" },
  { id: "invert", name: "INVERT" },
  { id: "invertSoft", name: "INVERT SOFT" },
];

// How often the random filter changes, in seconds.
const FILTER_SWITCH_SECONDS = 4;

// Filters work on a smaller copy of the video (this wide, in pixels)
// so a big shape doesn't slow the app down.
const FILTER_WORK_WIDTH = 480;

const COLOR_KEY = "fingerShapes.color";
const FILTERS_ON_KEY = "fingerShapes.randomFilters";
const OLD_SWAP_KEY = "fingerShapes.swap";

const stage = document.getElementById("stage");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const filterBtn = document.getElementById("filterBtn");
const colorInput = document.getElementById("colorInput");
const statusEl = document.getElementById("status");

let markColor = DEFAULT_COLOR;
let filtersOn = false;
let filterIndex = 0;
let filterTimer = null;

let stream = null;
let handLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let lastHandCount = -1;
let smoothedHands = [];
let fingerStates = [];
let fpsFrames = 0;
let fpsStart = 0;

// Remembers which fingertip each finger was connected to when its partner
// on the other hand wasn't raised, so the line doesn't jump back and forth.
let rungMemory = { a: new Map(), b: new Map() };

function resetRungMemory() {
  rungMemory = { a: new Map(), b: new Map() };
}

// A tiny tile of faint dark lines, repeated to make a scanline texture.
const scanlineTile = document.createElement("canvas");
scanlineTile.width = 1;
scanlineTile.height = 4;
const tileCtx = scanlineTile.getContext("2d");
tileCtx.fillStyle = "rgba(0, 0, 0, 0.14)";
tileCtx.fillRect(0, 0, 1, 1);
const scanlinePattern = ctx.createPattern(scanlineTile, "repeat");

// Small hidden canvases used by the filters.
const workCanvas = document.createElement("canvas");
const workCtx = workCanvas.getContext("2d", { willReadFrequently: true });
const tinyCanvas = document.createElement("canvas");
const tinyCtx = tinyCanvas.getContext("2d");
tinyCtx.imageSmoothingQuality = "high";

function setStatus(message) {
  statusEl.textContent = message;
  statusEl.hidden = false;
}

function friendlyError(error) {
  switch (error.name) {
    case "NotAllowedError":
      return "Camera is blocked. In Windows, open Settings > Privacy & security > Camera and turn on \"Let desktop apps access your camera\".";
    case "NotFoundError":
      return "No camera was found on this laptop.";
    case "NotReadableError":
      return "Your camera is being used by another app. Close it and try again.";
    default:
      return "Could not start the camera. Please try again.";
  }
}

// ---------- Settings (remembered on this laptop) ----------

function loadSetting(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    return null;
  }
}

function saveSetting(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    // Not saving is fine, the app still works.
  }
}

function removeSetting(key) {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    // Nothing to remove.
  }
}

function applyColor(color) {
  markColor = color;
  colorInput.value = color;
}

// Picks a random filter that is different from the current one.
function pickRandomFilter() {
  let next = filterIndex;
  while (next === filterIndex && FILTERS.length > 1) {
    next = Math.floor(Math.random() * FILTERS.length);
  }
  filterIndex = next;

  // Debug helper for the Console. We remove it in a later commit.
  window.latestFilter = FILTERS[filterIndex].id;
}

// The filter control is a one-click switch. On = a random filter that
// changes every few seconds. Off = no filter.
function setFiltersOn(on) {
  filtersOn = on;
  clearInterval(filterTimer);
  filterTimer = null;

  if (on) {
    filterIndex = -1;
    pickRandomFilter();
    filterTimer = setInterval(pickRandomFilter, FILTER_SWITCH_SECONDS * 1000);
  } else {
    window.latestFilter = "off";
  }

  filterBtn.setAttribute("aria-pressed", String(on));
}

function currentFilter() {
  return filtersOn ? FILTERS[filterIndex] : null;
}

// ---------- Smoothing ----------

// Moves each point a little toward its new position so the dots don't shake.
// Still hands get strong smoothing, fast hands get almost none.
// Hands are matched to the previous frame by wrist position and sorted
// left to right, so hand 1 and hand 2 keep a stable order.
// A hand that goes missing is kept for a few frames before it is dropped.
function smoothHands(rawHands) {
  const previous = smoothedHands;
  const used = new Set();

  const next = rawHands.map((hand) => {
    let best = -1;
    let bestDistance = 0.3;

    previous.forEach((prev, i) => {
      if (used.has(i)) return;
      const d = Math.hypot(hand[0].x - prev[0].x, hand[0].y - prev[0].y);
      if (d < bestDistance) {
        bestDistance = d;
        best = i;
      }
    });

    if (best === -1) {
      return hand.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    }

    used.add(best);
    const prev = previous[best];

    return hand.map((p, i) => {
      const speed = Math.hypot(p.x - prev[i].x, p.y - prev[i].y);
      const alpha = Math.min(1, 0.3 + speed * 15);
      return {
        x: prev[i].x + (p.x - prev[i].x) * alpha,
        y: prev[i].y + (p.y - prev[i].y) * alpha,
        z: p.z,
      };
    });
  });

  // Keep hands that went missing for a short moment.
  previous.forEach((prev, i) => {
    if (used.has(i)) return;
    const missed = (prev.missed || 0) + 1;
    if (missed <= HOLD_FRAMES) {
      const kept = prev.map((p) => ({ x: p.x, y: p.y, z: p.z }));
      kept.missed = missed;
      next.push(kept);
    }
  });

  next.sort((a, b) => a[0].x - b[0].x);
  smoothedHands = next;
  return next;
}

// ---------- Finger gestures ----------

function pixelDistance(a, b) {
  return Math.hypot(
    (a.x - b.x) * video.videoWidth,
    (a.y - b.y) * video.videoHeight
  );
}

function fingerRatio(hand, finger) {
  const base = hand[finger.base];
  const tipDistance = pixelDistance(hand[finger.tip], base);
  const jointDistance = Math.max(pixelDistance(hand[finger.joint], base), 1e-6);
  return tipDistance / jointDistance;
}

// Works out which fingers are out (true) or curled (false) on each hand.
function updateFingerStates(hands) {
  if (fingerStates.length !== hands.length) {
    fingerStates = hands.map(() => []);
  }

  const ratios = hands.map((hand, h) =>
    FINGERS.map((finger, f) => {
      const ratio = fingerRatio(hand, finger);
      const wasOut = fingerStates[h][f];
      fingerStates[h][f] = wasOut ? ratio > finger.off : ratio > finger.on;
      return ratio;
    })
  );

  // Debug helpers for the Console. We remove them in a later commit.
  // Order: thumb, index, middle, ring, pinky.
  window.latestFingers = fingerStates.map((states) => states.slice());
  window.latestRatios = ratios.map((row) =>
    row.map((value) => Math.round(value * 100) / 100)
  );
}

// ---------- Filters ----------

// Pixelate: shrink the picture, then blow it back up without smoothing.
function pixelate(w, h) {
  const block = Math.max(6, Math.round(w / 36));
  const tw = Math.max(1, Math.round(w / block));
  const th = Math.max(1, Math.round(h / block));
  tinyCanvas.width = tw;
  tinyCanvas.height = th;
  tinyCtx.imageSmoothingQuality = "high";
  tinyCtx.drawImage(workCanvas, 0, 0, tw, th);

  workCtx.save();
  workCtx.imageSmoothingEnabled = false;
  workCtx.clearRect(0, 0, w, h);
  workCtx.drawImage(tinyCanvas, 0, 0, tw, th, 0, 0, w, h);
  workCtx.restore();
}

// Glitch: random horizontal bands slide sideways, and the red and blue
// channels are pushed apart. The bands change every few frames.
let glitchBands = [];
let glitchTick = 0;

function makeGlitchBands(w, h) {
  const count = 3 + Math.floor(Math.random() * 3);
  glitchBands = Array.from({ length: count }, () => {
    const bandHeight = Math.max(2, Math.round(h * (0.02 + Math.random() * 0.08)));
    return {
      y: Math.floor(Math.random() * Math.max(1, h - bandHeight)),
      h: bandHeight,
      shift: Math.round((Math.random() - 0.5) * w * 0.25),
    };
  });
}

function glitch(w, h) {
  const image = workCtx.getImageData(0, 0, w, h);
  const src = image.data;
  const out = new Uint8ClampedArray(src);
  const split = Math.max(2, Math.round(w / 90));

  // Push red to the right and blue to the left.
  for (let y = 0; y < h; y++) {
    const rowStart = y * w;
    for (let x = 0; x < w; x++) {
      const i = (rowStart + x) * 4;
      const xr = Math.min(w - 1, x + split);
      const xb = Math.max(0, x - split);
      out[i] = src[(rowStart + xr) * 4];
      out[i + 2] = src[(rowStart + xb) * 4 + 2];
    }
  }

  // Slide a few bands sideways.
  if (glitchTick % 3 === 0 || glitchBands.length === 0) makeGlitchBands(w, h);
  glitchTick++;

  for (const band of glitchBands) {
    for (let y = band.y; y < band.y + band.h && y < h; y++) {
      const rowStart = y * w * 4;
      const row = out.slice(rowStart, rowStart + w * 4);
      for (let x = 0; x < w; x++) {
        const from = Math.min(w - 1, Math.max(0, x - band.shift)) * 4;
        const to = rowStart + x * 4;
        out[to] = row[from];
        out[to + 1] = row[from + 1];
        out[to + 2] = row[from + 2];
        out[to + 3] = row[from + 3];
      }
    }
  }

  workCtx.putImageData(new ImageData(out, w, h), 0, 0);
}

function shapeBounds(points) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(canvas.width, Math.ceil(Math.max(...xs)));
  const y1 = Math.min(canvas.height, Math.ceil(Math.max(...ys)));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// Draws the pixelated or glitched video for the area around the shape.
// The caller has already clipped the canvas to the area that gets the filter.
function drawFilteredVideo(filterId, corners, t) {
  const box = shapeBounds(corners);
  if (box.w < 2 || box.h < 2) return;

  const k = Math.min(1, FILTER_WORK_WIDTH / box.w);
  const w = Math.max(1, Math.round(box.w * k));
  const h = Math.max(1, Math.round(box.h * k));
  workCanvas.width = w;
  workCanvas.height = h;

  // Draw the video onto the small canvas, lined up with the shape's box.
  workCtx.drawImage(
    video,
    (t.offsetX - box.x) * k,
    (t.offsetY - box.y) * k,
    t.vw * t.scale * k,
    t.vh * t.scale * k
  );

  if (filterId === "pixelate") pixelate(w, h);
  if (filterId === "glitch") glitch(w, h);

  // Put the small canvas back over the shape's box.
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(workCanvas, box.x, box.y, box.w, box.h);
  ctx.restore();
}

// ---------- Canvas drawing ----------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(stage.clientWidth * dpr);
  canvas.height = Math.round(stage.clientHeight * dpr);
}

// The video fills the window with object-fit: cover, which crops it.
// This works out where the video sits so shapes land on the real fingertips.
function getCoverTransform() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.max(canvas.width / vw, canvas.height / vh);
  return {
    vw,
    vh,
    scale,
    offsetX: (canvas.width - vw * scale) / 2,
    offsetY: (canvas.height - vh * scale) / 2,
  };
}

function toCanvas(point, t) {
  return {
    x: t.offsetX + point.x * t.vw * t.scale,
    y: t.offsetY + point.y * t.vh * t.scale,
  };
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ---------- Lines and sections ----------

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function closestTip(from, list) {
  return list.reduce((best, tip) =>
    distance(from.p, tip.p) < distance(from.p, best.p) ? tip : best
  );
}

// Picks the fingertip on the other hand that a finger connects to:
//  - the same finger, if it is raised,
//  - otherwise the closest raised fingertip. To stop the line from jumping
//    between two fingertips that are almost the same distance, it keeps
//    its last target unless another one is clearly closer.
function pickTarget(from, others, memory) {
  const facing = others.find((o) => o.f === from.f);
  if (facing) {
    memory.set(from.f, facing.f);
    return facing;
  }

  const closest = closestTip(from, others);
  const kept = others.find((o) => o.f === memory.get(from.f));
  if (kept && distance(from.p, kept.p) <= distance(from.p, closest.p) * 1.25) {
    return kept;
  }

  memory.set(from.f, closest.f);
  return closest;
}

// One line (a "rung") for every raised finger. Each line goes from one hand
// to the other, never between two fingers of the same hand.
// handA and handB are lists of { f: finger number, p: fingertip position }.
function buildRungs(handA, handB) {
  const rungs = [];
  const seen = new Set();

  const add = (a, b) => {
    const key = `${a.f}-${b.f}`;
    if (seen.has(key)) return;
    seen.add(key);
    rungs.push({ a, b });
  };

  for (const a of handA) add(a, pickTarget(a, handB, rungMemory.a));
  for (const b of handB) add(pickTarget(b, handA, rungMemory.b), b);

  // Thumb side first, pinky side last.
  rungs.sort((r1, r2) => r1.a.f - r2.a.f || r1.b.f - r2.b.f);
  return rungs;
}

// Adds a triangle, turned the same way round as all the others, so the
// sections add up (and cancel in overlaps) the way we want.
function addTriangle(list, p, q, r) {
  const area = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  if (Math.abs(area) < 1) return;
  list.push(area > 0 ? [p, q, r] : [p, r, q]);
}

// The filled area is what lies between two neighboring lines, like the
// space between two rungs of a ladder. Each space is cut into two triangles.
function buildSections(rungs) {
  const sections = [];
  for (let i = 0; i < rungs.length - 1; i++) {
    const a1 = rungs[i].a.p;
    const b1 = rungs[i].b.p;
    const a2 = rungs[i + 1].a.p;
    const b2 = rungs[i + 1].b.p;
    addTriangle(sections, a1, b1, b2);
    addTriangle(sections, a1, b2, a2);
  }
  return sections;
}

function makeShape(tips) {
  const rungs = buildRungs(tips[0], tips[1]);
  const corners = rungs.flatMap((r) => [r.a.p, r.b.p]);
  return { rungs, corners, sections: buildSections(rungs) };
}

// All sections as one path. With the "nonzero" rule it covers every spot
// that any section touches. With the "evenodd" rule it covers only the spots
// touched by an odd number of sections, so overlaps are left out.
function sectionsPath(sections) {
  const path = new Path2D();
  for (const section of sections) {
    path.moveTo(section[0].x, section[0].y);
    for (let i = 1; i < section.length; i++) {
      path.lineTo(section[i].x, section[i].y);
    }
    path.closePath();
  }
  return path;
}

// Debug helper: roughly what percent of the shape is overlapped.
function overlapPercent(path, corners) {
  const box = shapeBounds(corners);
  const steps = 24;
  let covered = 0;
  let overlapped = 0;

  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const x = box.x + ((i + 0.5) * box.w) / steps;
      const y = box.y + ((j + 0.5) * box.h) / steps;
      if (!ctx.isPointInPath(path, x, y, "nonzero")) continue;
      covered++;
      if (!ctx.isPointInPath(path, x, y, "evenodd")) overlapped++;
    }
  }

  return covered ? Math.round((overlapped * 100) / covered) : 0;
}

// The outline goes along hand A's fingertips (thumb side to pinky side),
// across the last line, back along hand B's fingertips, and across the
// first line. It follows whatever shape the raised fingers make.
function outlinePoints(rungs) {
  return [...rungs.map((r) => r.a.p), ...rungs.map((r) => r.b.p).reverse()];
}

// The white lines: thin ones inside, a thicker outline around the shape.
function drawLines(rungs, dpr) {
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
  ctx.shadowBlur = 3 * dpr;

  // Thin lines: every line except the first and last, which are part of
  // the outline.
  if (rungs.length > 2) {
    ctx.lineWidth = INNER_LINE_WIDTH * dpr;
    ctx.beginPath();
    for (let i = 1; i < rungs.length - 1; i++) {
      ctx.moveTo(rungs[i].a.p.x, rungs[i].a.p.y);
      ctx.lineTo(rungs[i].b.p.x, rungs[i].b.p.y);
    }
    ctx.stroke();
  }

  // Thick outline.
  const ring = outlinePoints(rungs);
  ctx.lineWidth = OUTLINE_WIDTH * dpr;
  ctx.beginPath();
  ctx.moveTo(ring[0].x, ring[0].y);
  for (let i = 1; i < ring.length; i++) {
    ctx.lineTo(ring[i].x, ring[i].y);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.restore();
}

// The hand skeleton: thin lines between the joints and a small dot on
// every joint. The palm is always drawn. A closed finger is left out when
// HIDE_CLOSED_FINGERS is on. Raised fingertips that get a line also get a
// bigger dot with a white ring.
function drawHands(hands, t, dpr) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 3 * dpr;

  hands.forEach((hand, h) => {
    const pts = hand.map((p) => toCanvas(p, t));
    const bones = [...PALM_BONES];
    const joints = [...PALM_JOINTS];

    FINGER_PARTS.forEach((part, f) => {
      const raised = fingerStates[h] && fingerStates[h][f];
      if (HIDE_CLOSED_FINGERS && !raised) return;
      bones.push(...part.bones);
      joints.push(...part.joints);
    });

    // Bones.
    ctx.globalAlpha = 0.75;
    ctx.strokeStyle = markColor;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (const [a, b] of bones) {
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
    }
    ctx.stroke();

    // Joints.
    ctx.globalAlpha = 1;
    ctx.fillStyle = markColor;
    for (const i of joints) {
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, 3 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    // Bigger dots on the raised fingertips.
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2 * dpr;
    FINGERS.forEach((finger, f) => {
      const isCorner =
        fingerStates[h] && fingerStates[h][f] && SHAPE_FINGERS.includes(finger.name);
      if (!isCorner) return;
      const p = pts[finger.tip];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  });

  ctx.restore();
}

function drawOverlay(hands) {
  clearCanvas();
  window.latestRungs = 0;
  window.latestRungList = [];
  window.latestSections = 0;
  window.latestOverlap = 0;
  if (!video.videoWidth) return;

  const t = getCoverTransform();
  const dpr = window.devicePixelRatio || 1;

  // The raised fingertips of each hand, with their finger number
  // (0 thumb, 1 index, 2 middle, 3 ring, 4 pinky).
  const tips = hands.map((hand, h) =>
    FINGERS.flatMap((finger, f) =>
      fingerStates[h][f] && SHAPE_FINGERS.includes(finger.name)
        ? [{ f, p: toCanvas(hand[finger.tip], t) }]
        : []
    )
  );

  const bothHandsActive = tips.length === 2 && tips.every((list) => list.length > 0);

  if (bothHandsActive) {
    const shape = makeShape(tips);

    // Debug helpers for the Console. We remove them in a later commit.
    window.latestRungs = shape.rungs.length;
    window.latestRungList = shape.rungs.map(
      (r) => `${FINGERS[r.a.f].name}-${FINGERS[r.b.f].name}`
    );
    window.latestSections = shape.sections.length;

    // The area between the lines needs at least two lines.
    if (shape.sections.length > 0) {
      const path = sectionsPath(shape.sections);
      window.latestOverlap = overlapPercent(path, shape.corners);

      // The filter is drawn only where the shape is one layer thick, so
      // wherever two sections overlap, the plain picture shows through.
      const filter = currentFilter();
      if (filter) {
        ctx.save();
        ctx.clip(path, OVERLAP_KEEPS_FILTER ? "nonzero" : "evenodd");

        if (filter.id === "pixelate" || filter.id === "glitch") {
          drawFilteredVideo(filter.id, shape.corners, t);
        } else {
          ctx.filter =
            filter.id === "invert" ? "invert(1)" : "invert(1) hue-rotate(180deg)";
          ctx.drawImage(video, t.offsetX, t.offsetY, t.vw * t.scale, t.vh * t.scale);
          ctx.filter = "none";
        }

        ctx.restore();
      }

      // Faint scanlines over the whole shape.
      if (SHOW_SCANLINES && scanlinePattern) {
        ctx.save();
        ctx.clip(path, "nonzero");
        ctx.fillStyle = scanlinePattern;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
      }
    }

    drawLines(shape.rungs, dpr);
  }

  // The hand skeletons go on top of the shape.
  drawHands(hands, t, dpr);
}

new ResizeObserver(resizeCanvas).observe(stage);
resizeCanvas();

// ---------- Hand tracker ----------

async function createHandLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: TRACKING_SENSITIVITY,
    minHandPresenceConfidence: TRACKING_SENSITIVITY,
    minTrackingConfidence: TRACKING_SENSITIVITY,
  });
}

async function loadHandTracker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    try {
      handLandmarker = await createHandLandmarker(vision, "GPU");
      console.log("Hand tracker ready (GPU)");
    } catch (gpuError) {
      console.warn("GPU not available, using CPU instead.", gpuError);
      handLandmarker = await createHandLandmarker(vision, "CPU");
      console.log("Hand tracker ready (CPU)");
    }
    startTracking();
  } catch (error) {
    console.error("Hand tracker failed to load:", error);
  }
}

function trackLoop() {
  if (!stream || !handLandmarker) {
    animationId = null;
    return;
  }

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());

    const hands = smoothHands(result.landmarks);

    // Debug helper for the Console. We remove it in a later commit.
    window.latestHands = hands;

    const count = hands.length;
    if (count !== lastHandCount) {
      lastHandCount = count;
      console.log("Hands found:", count);
    }

    updateFingerStates(hands);
    drawOverlay(hands);

    // Debug helper: frames per second, updated once a second.
    fpsFrames++;
    const now = performance.now();
    if (now - fpsStart >= 1000) {
      window.latestFps = Math.round((fpsFrames * 1000) / (now - fpsStart));
      fpsFrames = 0;
      fpsStart = now;
    }
  }

  animationId = requestAnimationFrame(trackLoop);
}

function startTracking() {
  if (animationId !== null || !stream || !handLandmarker) return;
  lastVideoTime = -1;
  lastHandCount = -1;
  smoothedHands = [];
  fingerStates = [];
  resetRungMemory();
  fpsFrames = 0;
  fpsStart = performance.now();
  animationId = requestAnimationFrame(trackLoop);
}

function stopTracking() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  smoothedHands = [];
  fingerStates = [];
  resetRungMemory();
  window.latestHands = [];
  window.latestRungs = 0;
  window.latestRungList = [];
  window.latestSections = 0;
  window.latestOverlap = 0;
  window.latestFingers = [];
  window.latestRatios = [];
  window.latestFps = 0;
  clearCanvas();
}

// ---------- Camera ----------

function stopCamera() {
  stopTracking();
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.hidden = true;
  canvas.hidden = true;
  startBtn.title = "Start camera";
  startBtn.setAttribute("aria-label", "Start camera");
  setStatus("Camera is off");
}

async function startCamera() {
  startBtn.disabled = true;
  setStatus("Starting camera...");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    video.hidden = false;
    canvas.hidden = false;
    statusEl.hidden = true;
    startBtn.title = "Stop camera";
    startBtn.setAttribute("aria-label", "Stop camera");

    startTracking();
  } catch (error) {
    console.error("Camera error:", error);
    stopCamera();
    setStatus(friendlyError(error));
  } finally {
    startBtn.disabled = false;
  }
}

// ---------- Buttons ----------

startBtn.addEventListener("click", () => {
  if (stream) {
    stopCamera();
  } else {
    startCamera();
  }
});

filterBtn.addEventListener("click", () => {
  setFiltersOn(!filtersOn);
  saveSetting(FILTERS_ON_KEY, String(filtersOn));
});

colorInput.addEventListener("input", () => {
  applyColor(colorInput.value);
  saveSetting(COLOR_KEY, colorInput.value);
});

// ---------- Start ----------

const savedColor = loadSetting(COLOR_KEY);
applyColor(/^#[0-9a-f]{6}$/i.test(savedColor || "") ? savedColor : DEFAULT_COLOR);
setFiltersOn(loadSetting(FILTERS_ON_KEY) === "true");
removeSetting(OLD_SWAP_KEY);

startBtn.disabled = false;
console.log("Finger Shapes loaded");
loadHandTracker();