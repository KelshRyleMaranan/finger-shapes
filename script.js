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

// Which fingers can make corners. Remove a name to stop that finger
// from ever counting, for example remove "ring" and "pinky" for fewer corners.
const SHAPE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];

// The lines of the hand skeleton, as pairs of the 21 MediaPipe points.
const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// Camcorder yellow. The person can change it with the color circle.
const DEFAULT_COLOR = "#ffd21f";

// Filters that can be applied inside the shape. When the filter button is
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
const SWAP_KEY = "fingerShapes.swap";
const FILTERS_ON_KEY = "fingerShapes.randomFilters";

const stage = document.getElementById("stage");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const swapBtn = document.getElementById("swapBtn");
const filterBtn = document.getElementById("filterBtn");
const colorInput = document.getElementById("colorInput");
const statusEl = document.getElementById("status");
const toastEl = document.getElementById("toast");

let markColor = DEFAULT_COLOR;
let swapped = false;
let filtersOn = false;
let filterIndex = 0;
let filterTimer = null;
let toastTimer = null;

let stream = null;
let handLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let lastHandCount = -1;
let smoothedHands = [];
let fingerStates = [];
let fpsFrames = 0;
let fpsStart = 0;

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

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1600);
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

function applyColor(color) {
  markColor = color;
  colorInput.value = color;
  document.documentElement.style.setProperty("--mark", color);
}

function applySwap(on) {
  swapped = on;
  stage.classList.toggle("swapped", on);
  swapBtn.setAttribute("aria-pressed", String(on));
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

// The filter button is a one-click switch. On = a random filter that
// changes every few seconds. Off = the normal look.
function setFiltersOn(on, announce) {
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
  filterBtn.setAttribute("aria-label", "Random filters");
  filterBtn.title = `Random filters: ${on ? "ON" : "OFF"}`;
  if (announce) showToast(on ? "FILTERS: RANDOM" : "FILTERS: OFF");
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

function shapeBounds(shape) {
  const xs = shape.map((p) => p.x);
  const ys = shape.map((p) => p.y);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(canvas.width, Math.ceil(Math.max(...xs)));
  const y1 = Math.min(canvas.height, Math.ceil(Math.max(...ys)));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// Draws the pixelated or glitched video for the area around the shape.
// The caller has already clipped the canvas to the shape.
function drawFilteredVideo(filterId, shape, t) {
  const box = shapeBounds(shape);
  if (box.w < 2 || box.h < 2) return;

  const k = Math.min(1, FILTER_WORK_WIDTH / box.w);
  const w = Math.max(1, Math.round(box.w * k));
  const h = Math.max(1, Math.round(box.h * k));
  workCanvas.width = w;
  workCanvas.height = h;

  // Draw the video onto the small canvas, lined up with the shape's box.
  workCtx.save();
  workCtx.filter = swapped ? "grayscale(1) contrast(1.25)" : "none";
  workCtx.drawImage(
    video,
    (t.offsetX - box.x) * k,
    (t.offsetY - box.y) * k,
    t.vw * t.scale * k,
    t.vh * t.scale * k
  );
  workCtx.restore();

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

// Sorts points by angle around their center so the outline never twists.
function orderAroundCenter(points) {
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  return points
    .slice()
    .sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
}

function traceShape(shape) {
  ctx.beginPath();
  ctx.moveTo(shape[0].x, shape[0].y);
  for (let i = 1; i < shape.length; i++) {
    ctx.lineTo(shape[i].x, shape[i].y);
  }
  ctx.closePath();
}

// Thin white outline around the shape.
function drawOutline(shape, dpr) {
  ctx.save();
  traceShape(shape);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
  ctx.lineWidth = 2 * dpr;
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
  ctx.shadowBlur = 3 * dpr;
  ctx.stroke();
  ctx.restore();
}

// The hand skeleton: thin lines between the 21 points and a small dot on
// every joint. Fingertips that count as corners get a bigger dot with a
// white ring, so you can see which fingers make the shape.
function drawHands(hands, t, dpr) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 3 * dpr;

  hands.forEach((hand, h) => {
    const pts = hand.map((p) => toCanvas(p, t));

    // Bones.
    ctx.globalAlpha = 0.75;
    ctx.strokeStyle = markColor;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (const [a, b] of HAND_BONES) {
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
    }
    ctx.stroke();

    // Joints.
    ctx.globalAlpha = 1;
    ctx.fillStyle = markColor;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    // Bigger dots on the fingertips that are corners of the shape.
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

// Size label above the shape, like "MONO PIXELATE 487x293".
// The canvas is mirrored by CSS, so the text is flipped here to read correctly.
function drawLabel(shape, t, dpr) {
  const xs = shape.map((p) => p.x);
  const ys = shape.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const width = Math.round((maxX - minX) / t.scale);
  const height = Math.round((maxY - minY) / t.scale);
  const filter = currentFilter();
  const filterText = filter ? ` ${filter.name}` : "";
  const text = `${swapped ? "MONO" : "COLOR"}${filterText} ${width}x${height}`;

  const fontSize = 11 * dpr;

  ctx.save();
  ctx.font = `bold ${fontSize}px Consolas, "Courier New", monospace`;
  const textWidth = ctx.measureText(text).width;

  // On screen the label starts at the left edge of the shape (canvas maxX).
  const x = Math.min(Math.max(maxX, textWidth + 4 * dpr), canvas.width - 4 * dpr);
  let y = minY - 8 * dpr;
  if (y < fontSize + 4 * dpr) y = minY + fontSize + 8 * dpr;

  ctx.translate(x, y);
  ctx.scale(-1, 1);
  ctx.lineJoin = "round";
  ctx.lineWidth = 3 * dpr;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
  ctx.strokeText(text, 0, 0);
  ctx.fillStyle = markColor;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawOverlay(hands) {
  clearCanvas();
  window.latestShape = [];
  if (!video.videoWidth) return;

  const t = getCoverTransform();
  const dpr = window.devicePixelRatio || 1;

  // Only fingers that are out become corners.
  const points = hands.map((hand, h) =>
    FINGERS.flatMap((finger, f) =>
      fingerStates[h][f] && SHAPE_FINGERS.includes(finger.name)
        ? [toCanvas(hand[finger.tip], t)]
        : []
    )
  );

  const total = points.flat().length;
  const bothHandsActive = points.length === 2 && points.every((p) => p.length > 0);

  let shape = null;

  if (bothHandsActive && total >= 3) {
    shape = orderAroundCenter(points.flat());
    window.latestShape = shape;

    // Inside the shape we draw the video again, clipped to the shape.
    // Normal: the video behind is gray, so this draws it in color.
    // Swapped: the video behind is in color, so this draws it gray with extra
    // contrast. A filter, if one is on, changes the picture inside.
    // Faint scanlines go on top in every mode.
    const filter = currentFilter();
    const filterId = filter ? filter.id : "none";

    ctx.save();
    traceShape(shape);
    ctx.clip();

    if (filterId === "pixelate" || filterId === "glitch") {
      drawFilteredVideo(filterId, shape, t);
    } else {
      const parts = [];
      if (swapped) parts.push("grayscale(1) contrast(1.25)");
      if (filterId === "invert") parts.push("invert(1)");
      if (filterId === "invertSoft") parts.push("invert(1) hue-rotate(180deg)");
      ctx.filter = parts.length ? parts.join(" ") : "none";
      ctx.drawImage(video, t.offsetX, t.offsetY, t.vw * t.scale, t.vh * t.scale);
      ctx.filter = "none";
    }

    if (scanlinePattern) {
      ctx.fillStyle = scanlinePattern;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.restore();

    drawOutline(shape, dpr);
  }

  // The hand skeletons go on top of the shape.
  drawHands(hands, t, dpr);

  if (shape) drawLabel(shape, t, dpr);
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
  window.latestHands = [];
  window.latestShape = [];
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

swapBtn.addEventListener("click", () => {
  applySwap(!swapped);
  saveSetting(SWAP_KEY, String(swapped));
});

filterBtn.addEventListener("click", () => {
  setFiltersOn(!filtersOn, true);
  saveSetting(FILTERS_ON_KEY, String(filtersOn));
});

colorInput.addEventListener("input", () => {
  applyColor(colorInput.value);
  saveSetting(COLOR_KEY, colorInput.value);
});

// ---------- Start ----------

const savedColor = loadSetting(COLOR_KEY);
applyColor(/^#[0-9a-f]{6}$/i.test(savedColor || "") ? savedColor : DEFAULT_COLOR);
applySwap(loadSetting(SWAP_KEY) === "true");
setFiltersOn(loadSetting(FILTERS_ON_KEY) === "true", false);

startBtn.disabled = false;
console.log("Finger Shapes loaded");
loadHandTracker();