async function loadModel() {
  try {
    document.getElementById("status").innerText = "⏳ Loading model...";
    
    await tf.setBackend('wasm');
    await tf.ready();
    console.log("✅ TFJS ready with backend:", tf.getBackend());
    const modelUrl = chrome.runtime.getURL("model/model.json");
    const model = await tf.loadLayersModel(modelUrl);
    document.getElementById("status").innerText = "✅ Model loaded successfully!";
    return model;
  } catch (err) {
    console.error("❌ Error loading model:", err);
    document.getElementById("status").innerText = "❌ Failed to load model.";
    return null;
  }
}

async function classifyImage(model, imageElement) {
  try {
    const imgTensor = tf.browser.fromPixels(imageElement)
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat()
      .div(tf.scalar(255.0));

    const prediction = model.predict(imgTensor);
    const score = (await prediction.data())[0];
    prediction.dispose();
    imgTensor.dispose();

    return score;
  } catch (error) {
    console.error("⚠️ Classification error:", error);
    return null;
  }
}

async function runModelOnPage() {
  document.getElementById("status").innerText = "🔍 Scanning webpage images...";
  
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const imgs = Array.from(document.images).map(img => img.src);
      return imgs;
    }
  }, async (results) => {
    const imageUrls = results[0].result;
    if (!imageUrls.length) {
      document.getElementById("status").innerText = "⚠️ No images found on page.";
      return;
    }

    const model = await loadModel();
    if (!model) return;

    document.getElementById("status").innerText = `🖼 Found ${imageUrls.length} images. Classifying...`;

    for (const src of imageUrls) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = src;

      await new Promise((resolve) => {
        img.onload = async () => {
          const score = await classifyImage(model, img);
          console.log(`🧠 Prediction for ${src}:`, score);
          resolve();
        };
        img.onerror = () => resolve();
      });
    }

    document.getElementById("status").innerText = "✅ Finished scanning images!";
  });
}

document.getElementById("runModel").addEventListener("click", runModelOnPage);
