// Следящая рамка — крайние точки тела (ладони, ступни, верх головы)
// задают границы кадра. MediaPipe Pose Landmarker, чистый JS, без сборки.

import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ── DOM ─────────────────────────────────────────────────────────
const video      = document.getElementById("video");
const canvas     = document.getElementById("canvas");
const canvasWrap = document.getElementById("canvasWrap");
const ctx        = canvas.getContext("2d");
const statusEl   = document.getElementById("status");

const btnCam    = document.getElementById("btnCam");
const fileInput = document.getElementById("fileInput");
const btnPlay   = document.getElementById("btnPlay");
const btnBg     = document.getElementById("btnBg");
const btnRec    = document.getElementById("btnRec");

const paddingInput   = document.getElementById("padding");
const smoothingInput = document.getElementById("smoothing");
const captionInput   = document.getElementById("caption");

// ── состояние ───────────────────────────────────────────────────
let landmarker = null;
let camStream  = null;
let rafId      = null;
let lastVideoTime = -1;

let box = null;        // сглаженная рамка {x, y, w, h}
let targetBox = null;  // цель текущего кадра

let bgWhite = true;
const BG = { white: "#f6f5f1", black: "#0d0d0d" };
const INK = { white: "#141414", black: "#f6f5f1" };

let recorder = null;
let recChunks = [];

// ── индексы точек MediaPipe Pose ────────────────────────────────
// ладони: запястья + пальцы; ступни: лодыжки, пятки, носки
const HAND_IDS = [15, 16, 17, 18, 19, 20, 21, 22];
const FOOT_IDS = [27, 28, 29, 30, 31, 32];
const NOSE = 0, L_EYE = 2, R_EYE = 5, L_EAR = 7, R_EAR = 8;
const MIN_VISIBILITY = 0.4;

// ── загрузка модели ─────────────────────────────────────────────
async function initLandmarker() {
  if (landmarker) return;
  setStatus("Загружаю модель позы…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

// ── источники ───────────────────────────────────────────────────
btnCam.addEventListener("click", async () => {
  try {
    await initLandmarker();
    stopSource();
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = camStream;
    video.loop = false;
    await startVideo();
  } catch (err) {
    setStatus("Не удалось запустить камеру: " + err.message);
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  try {
    await initLandmarker();
    stopSource();
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = true;
    await startVideo();
  } catch (err) {
    setStatus("Не удалось открыть файл: " + err.message);
  }
});

function stopSource() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  if (video.src) URL.revokeObjectURL(video.src);
  box = null;
  lastVideoTime = -1;
}

async function startVideo() {
  await video.play();
  await new Promise((res) => {
    if (video.videoWidth) res();
    else video.addEventListener("loadedmetadata", res, { once: true });
  });
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  statusEl.hidden = true;
  canvasWrap.hidden = false;
  btnPlay.disabled = false;
  btnRec.disabled = false;
  btnPlay.textContent = "Пауза";

  if (!rafId) loop();
}

// ── воспроизведение ─────────────────────────────────────────────
btnPlay.addEventListener("click", () => {
  if (video.paused) {
    video.play();
    btnPlay.textContent = "Пауза";
  } else {
    video.pause();
    btnPlay.textContent = "Играть";
  }
});

btnBg.addEventListener("click", () => {
  bgWhite = !bgWhite;
  btnBg.textContent = bgWhite ? "Фон: белый" : "Фон: чёрный";
});

// ── рамка по крайним точкам ─────────────────────────────────────
function computeTargetBox(lm, W, H) {
  const xs = [], ys = [];

  for (const i of [...HAND_IDS, ...FOOT_IDS]) {
    const p = lm[i];
    if (!p) continue;
    if (p.visibility !== undefined && p.visibility < MIN_VISIBILITY) continue;
    xs.push(p.x * W);
    ys.push(p.y * H);
  }

  // верх головы: от линии глаз вверх примерно на ширину головы
  const nose = lm[NOSE], le = lm[L_EYE], re = lm[R_EYE];
  const lear = lm[L_EAR], rear = lm[R_EAR];
  if (nose) {
    const eyeY = le && re ? ((le.y + re.y) / 2) * H : nose.y * H;
    const headW =
      lear && rear
        ? Math.hypot((lear.x - rear.x) * W, (lear.y - rear.y) * H)
        : H * 0.06;
    xs.push(nose.x * W);
    ys.push(eyeY - headW * 0.9);
  }

  if (xs.length < 3) return null;

  const pad = (parseFloat(paddingInput.value) / 100) * Math.min(W, H);
  const minX = Math.min(...xs) - pad;
  const maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad;
  const maxY = Math.max(...ys) + pad;

  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function lerpBox(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
    h: a.h + (b.h - a.h) * t,
  };
}

// ── основной цикл ───────────────────────────────────────────────
function loop() {
  rafId = requestAnimationFrame(loop);
  if (!landmarker || video.readyState < 2) return;

  if (!video.paused && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    if (result.landmarks && result.landmarks.length > 0) {
      const t = computeTargetBox(
        result.landmarks[0],
        canvas.width,
        canvas.height
      );
      if (t) targetBox = t;
    }
  }

  if (targetBox) {
    // плавность: 1 — мгновенно, 30 — очень вязко
    const alpha = 1 / parseFloat(smoothingInput.value);
    box = box ? lerpBox(box, targetBox, alpha) : { ...targetBox };
  }

  draw();
}

// ── отрисовка ───────────────────────────────────────────────────
function draw() {
  const W = canvas.width, H = canvas.height;
  const bg = bgWhite ? BG.white : BG.black;
  const ink = bgWhite ? INK.white : INK.black;

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  if (!box) return;

  const { x, y, w, h } = box;

  // видео только внутри рамки
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  // линии рамки — до краёв холста
  const lw = Math.max(1, W / 1100);
  ctx.strokeStyle = ink;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(0, y);         ctx.lineTo(W, y);
  ctx.moveTo(0, y + h);     ctx.lineTo(W, y + h);
  ctx.moveTo(x, 0);         ctx.lineTo(x, H);
  ctx.moveTo(x + w, 0);     ctx.lineTo(x + w, H);
  ctx.stroke();

  // угловые маркеры
  const s = Math.max(10, W * 0.014);
  ctx.fillStyle = bg;
  for (const [cx, cy] of [
    [x, y], [x + w, y], [x, y + h], [x + w, y + h],
  ]) {
    ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
    ctx.strokeRect(cx - s / 2, cy - s / 2, s, s);
  }

  // подпись по периметру
  const text = captionInput.value.trim();
  if (text) drawCaptions(text, x, y, w, h, ink, W);
}

function drawCaptions(text, x, y, w, h, ink, W) {
  const size = Math.max(11, W * 0.014);
  const gap = size * 0.7;
  ctx.fillStyle = ink;
  ctx.font = `${size}px "IBM Plex Mono", monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";

  // сверху — обычная
  ctx.fillText(text, x + w / 2, y - gap);

  // снизу — вверх ногами
  ctx.save();
  ctx.translate(x + w / 2, y + h + gap);
  ctx.rotate(Math.PI);
  ctx.fillText(text, 0, 0);
  ctx.restore();

  // слева — читается снизу вверх
  ctx.save();
  ctx.translate(x - gap, y + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(text, 0, 0);
  ctx.restore();

  // справа — читается сверху вниз
  ctx.save();
  ctx.translate(x + w + gap, y + h / 2);
  ctx.rotate(Math.PI / 2);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

// ── запись холста ───────────────────────────────────────────────
btnRec.addEventListener("click", () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  recorder = new MediaRecorder(stream, {
    mimeType: mime,
    videoBitsPerSecond: 12_000_000,
  });
  recChunks = [];
  recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(recChunks, { type: "video/webm" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tracking-frame.webm";
    a.click();
    URL.revokeObjectURL(a.href);
    btnRec.classList.remove("recording");
    btnRec.innerHTML = '<span class="dot"></span>Запись';
  };
  recorder.start();
  btnRec.classList.add("recording");
  btnRec.innerHTML = '<span class="dot"></span>Стоп';
});

// ── утилиты ─────────────────────────────────────────────────────
function setStatus(msg) {
  statusEl.hidden = false;
  statusEl.textContent = msg;
}
