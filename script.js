const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");

let stream = null;

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

function stopCamera() {
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