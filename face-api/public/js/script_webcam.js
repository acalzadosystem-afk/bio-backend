const video = document.getElementById('inputVideo');
const canvas = document.getElementById('overlay');

const MODEL_URL = '/public/models';

let useTiny = true;
let opts; // se define según el backend

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false
  });
  video.srcObject = stream;
  return new Promise(r => (video.onloadedmetadata = () => { video.play(); r(); }));
}

async function loadModels() {
  // Intenta tiny; si falta, cae a SSD
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  } catch (e) {
    useTiny = false;
  }

  if (useTiny) {
    // modelos extra (si los tienes)
    await Promise.all([
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
  } else {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await Promise.all([
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    opts = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
  }
}

function draw(results) {
  const dims = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(canvas, dims);
  const resized = faceapi.resizeResults(results, dims);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (Array.isArray(resized)) {
    faceapi.draw.drawDetections(canvas, resized);
    try { faceapi.draw.drawFaceLandmarks(canvas, resized); } catch {}
    try { faceapi.draw.drawFaceExpressions(canvas, resized, 0.05); } catch {}
  }
}

const INTERVAL_MS = 80;
const LANDMARK_EVERY = 3;
let frameCount = 0;

async function loop() {
  const t0 = performance.now();

  // detección ligera
  const detOnly = await faceapi.detectAllFaces(video, opts);
  draw(detOnly);

  // landmarks/expresiones cada N ciclos
  if (frameCount % LANDMARK_EVERY === 0 && detOnly.length > 0) {
    const rich = await faceapi
      .detectAllFaces(video, opts)
      .withFaceLandmarks()
      .withFaceExpressions();
    draw(rich);
  }

  frameCount++;
  const spent = performance.now() - t0;
  setTimeout(loop, Math.max(0, INTERVAL_MS - spent));
}

(async () => {
  try {
    if (typeof tf !== 'undefined' && tf.getBackend() !== 'webgl') {
      await tf.setBackend('webgl'); await tf.ready();
    }
    await setupCamera();
    await loadModels();
    loop();
  } catch (err) {
    alert('Error inicializando cámara o modelos:\n' + err);
    console.error(err);
  }
})();

