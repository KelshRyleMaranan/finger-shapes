import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");

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
}

function stopCamera() {
  stopTracking();
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.hidden = true;
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