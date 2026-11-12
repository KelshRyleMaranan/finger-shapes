import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// MediaPipe point numbers for the fingertips that make the shape:
// 4 = thumb, 8 = index. Add 12 (middle), 16 (ring), 20 (pinky) for more corners.
const SHAPE_FINGERTIPS = [4, 8];

const stage = document.getElementById("stage");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");

const accent =
  getComputedStyle(document.documentElement)
    .getPropertyValue("--accent")
    .trim() || "#e8892f";

let stream = null;
let handLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let lastHandCount = -1;
let smoothedHands = [];

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

// ---------- Smoothing ----------

// Moves each point a little toward its new position so the dots don't shake.
// Still hands get strong smoothing, fast hands get almost none.
// Hands are matched to the previous frame by wrist position and sorted
// left to right, so hand 1 and hand 2 keep a stable order.
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

  next.sort((a, b) => a[0].x - b[0].x);
  smoothedHands = next;
  return next;
}

// ---------- Canvas drawing ----------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(stage.clientWidth * dpr);
  canvas.height = Math.round(stage.clientHeight * dpr);
}

// The video fills the window with object-fit: cover, which crops it.
// This works out where the video sits so dots land on the real fingertips.
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

function drawOverlay(hands) {
  clearCanvas();
  window.latestShape = [];
  if (!video.videoWidth) return;

  const t = getCoverTransform();
  const dpr = window.devicePixelRatio || 1;

  // Chosen fingertip positions on the canvas, one list per hand.
  const points = hands.map((hand) =>
    SHAPE_FINGERTIPS.map((index) => toCanvas(hand[index], t))
  );

  if (points.length === 2) {
    const shape = orderAroundCenter(points.flat());
    window.latestShape = shape;

    // Normal color inside the shape: draw the plain video clipped to it.
    // The video behind the canvas is gray, so everything outside stays gray.
    ctx.save();
    traceShape(shape);
    ctx.clip();
    ctx.drawImage(video, t.offsetX, t.offsetY, t.vw * t.scale, t.vh * t.scale);
    ctx.restore();

    // Outline.
    traceShape(shape);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 3 * dpr;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  // Dots go last so they sit on top.
  ctx.fillStyle = accent;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2 * dpr;

  for (const handPoints of points) {
    for (const p of handPoints) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

new ResizeObserver(resizeCanvas).observe(stage);
resizeCanvas();

// ---------- Hand tracker ----------

async function createHandLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
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

    drawOverlay(hands);
  }

  animationId = requestAnimationFrame(trackLoop);
}

function startTracking() {
  if (animationId !== null || !stream || !handLandmarker) return;
  lastVideoTime = -1;
  lastHandCount = -1;
  smoothedHands = [];
  animationId = requestAnimationFrame(trackLoop);
}

function stopTracking() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  smoothedHands = [];
  window.latestHands = [];
  window.latestShape = [];
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

startBtn.addEventListener("click", () => {
  if (stream) {
    stopCamera();
  } else {
    startCamera();
  }
});

startBtn.disabled = false;
console.log("Finger Shapes loaded");
loadHandTracker();