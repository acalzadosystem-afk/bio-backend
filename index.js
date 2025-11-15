const express = require('express');
const cors = require('cors');
const path = require('path');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs'); // versión JS, SIN tfjs-node

// Hacemos que face-api use este tf
global.tf = tf;

// Parches para usar node-canvas como si fuera canvas del navegador
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' })); // para imágenes base64 grandes

const MODELS_PATH = path.join(__dirname, 'models');

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODELS_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_PATH);
  console.log('✅ Modelos de face-api cargados');
}

app.post('/api/recognize', async (req, res) => {
  try {
    const { image } = req.body;

    if (!image) {
      return res.status(400).json({ success: false, message: 'Falta imagen' });
    }

    // image viene en base64 (solo los datos, sin 'data:image/png;base64,')
    const imgBuffer = Buffer.from(image, 'base64');

    const img = await canvas.loadImage(imgBuffer);
    const c = canvas.createCanvas(img.width, img.height);
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);

    const options = new faceapi.TinyFaceDetectorOptions({
      inputSize: 320,
      scoreThreshold: 0.5,
    });

    const result = await faceapi
      .detectSingleFace(c, options)
      .withFaceLandmarks();

    if (!result) {
      return res.json({
        success: false,
        message: 'No se detectó rostro',
      });
    }

    const { box } = result.detection;
    const landmarks = result.landmarks;

    const width = img.width;
    const height = img.height;

    // Normalizamos a 0..1 para luego convertir a %
    const normBox = {
      x: box.x / width,
      y: box.y / height,
      width: box.width / width,
      height: box.height / height,
    };

    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();
    const nose = landmarks.getNose();

    const avgPoint = pts => ({
      x: pts.reduce((s, p) => s + p.x, 0) / pts.length / width,
      y: pts.reduce((s, p) => s + p.y, 0) / pts.length / height,
    });

    const payload = {
      success: true,
      box: normBox,
      landmarks: {
        leftEye: avgPoint(leftEye),
        rightEye: avgPoint(rightEye),
        nose: avgPoint(nose),
      },
      userId: null, // luego lo usamos para reconocimiento, ahora solo detección
    };

    return res.json(payload);
  } catch (err) {
    console.error('❌ Error en /api/recognize:', err);
    return res.status(500).json({ success: false, message: 'Error en servidor' });
  }
});

loadModels().then(() => {
  app.listen(3000, () => {
    console.log('🚀 Servidor escuchando en http://0.0.0.0:3000');
  });
});
