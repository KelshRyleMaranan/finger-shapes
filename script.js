import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// MediaPipe point numbers for the five fingertips:
// thumb, index, middle, ring, pinky.
const FINGERTIPS = [4, 8, 12, 16, 20];

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

function drawFingertips(hands) {
  clearCanvas();
  if (!video.videoWidth) return;

  const t = getCoverTransform();
  const dpr = window.devicePixelRatio || 1;

  ctx.fillStyle = accent;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2 * dpr;

  for (const hand of hands) {
    for (const index of FINGERTIPS) {
      const p = toCanvas(hand[index], t);
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

    // Debug helper for the Console. We remove it in a later commit.
    window.latestHands = result.landmarks;

    const count = result.landmarks.length;
    if (count !== lastHandCount) {
      lastHandCount = count;
      console.log("Hands found:", count);
    }

    drawFingertips(result.landmarks);
  }

  animationId = requestAnimationFrame(trackLoop);
}

function startTracking() {
  if (animationId !== null || !stream || !handLandmarker) return;
  lastVideoTime = -1;
  lastHandCount = -1;
  animationId = requestAnimationFrame(trackLoop);
}

function stopTracking() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  window.latestHands = [];
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