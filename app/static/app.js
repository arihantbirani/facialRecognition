const fileInput = document.getElementById("file-input");
const recognizeButton = document.getElementById("recognize-button");
const statusText = document.getElementById("status");
const previewShell = document.getElementById("preview-shell");
const imageStage = document.getElementById("image-stage");
const previewImage = document.getElementById("preview-image");
const overlay = document.getElementById("overlay");
const results = document.getElementById("results");
const resultCount = document.getElementById("result-count");
const supportedExtensions = [".jpg", ".jpeg", ".png", ".bmp"];
const apiBaseUrl = resolveApiBaseUrl();

let selectedFile = null;
let imageBitmap = null;
let objectUrl = null;
let currentMatches = [];

fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  selectedFile = file ?? null;
  currentMatches = [];
  renderResults();
  clearOverlay();

  if (!selectedFile) {
    resetPreview();
    setStatus("Select an image to begin.");
    return;
  }

  if (!isSupportedFile(selectedFile)) {
    resetSelection();
    setStatus(`Unsupported file type. Please use ${supportedExtensions.join(", ")}.`);
    return;
  }

  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }

  try {
    objectUrl = URL.createObjectURL(selectedFile);
    previewImage.src = objectUrl;
    previewImage.style.display = "block";
    previewShell.classList.remove("empty");
    imageBitmap = await createImageBitmap(selectedFile);
    await previewImage.decode();
    syncOverlaySize();
    setStatus("Image loaded. Run recognition to detect faces.");
  } catch (error) {
    resetSelection();
    setStatus("Unable to preview this image. Convert it to JPG or PNG and try again.");
  }
});

recognizeButton.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("Please select an image first.");
    return;
  }

  recognizeButton.disabled = true;
  setStatus("Running recognition...");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    const response = await fetch(`${apiBaseUrl}/recognize`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "recognition failed");
    }

    currentMatches = payload.matches || [];
    drawMatches(currentMatches);
    renderResults();
    setStatus(`Recognition complete. ${currentMatches.length} face(s) detected.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    recognizeButton.disabled = false;
  }
});

window.addEventListener("resize", () => {
  syncOverlaySize();
  drawMatches(currentMatches);
});

function syncOverlaySize() {
  if (!previewImage.naturalWidth) {
    return;
  }
  const width = previewImage.clientWidth;
  const height = previewImage.clientHeight;
  imageStage.style.width = `${width}px`;
  imageStage.style.height = `${height}px`;
  overlay.width = width;
  overlay.height = height;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
}

function clearOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawMatches(matches) {
  syncOverlaySize();
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!previewImage.naturalWidth || !matches.length) {
    return;
  }

  const scaleX = previewImage.clientWidth / previewImage.naturalWidth;
  const scaleY = previewImage.clientHeight / previewImage.naturalHeight;

  ctx.lineWidth = 3;
  ctx.font = "15px Georgia";

  matches.forEach((match, index) => {
    const [x1, y1, x2, y2] = match.box;
    const color = match.is_known ? "#2d6a4f" : "#9d2b2b";
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x, y, width, height);
    ctx.fillRect(x, Math.max(0, y - 24), 160, 24);
    ctx.fillStyle = "#fff";
    ctx.fillText(`${index + 1}. ${match.identity}`, x + 8, Math.max(16, y - 8));
  });
}

function renderResults() {
  results.innerHTML = "";
  resultCount.textContent = `${currentMatches.length} detection${currentMatches.length === 1 ? "" : "s"}`;

  if (!currentMatches.length) {
    results.innerHTML = `<p class="meta">No recognition results available yet.</p>`;
    return;
  }

  currentMatches.forEach((match, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const identityClass = match.is_known ? "identity-known" : "identity-unknown";
    const displayIdentity = match.is_known ? match.identity : "Unknown";
    card.innerHTML = `
      <h3>Detection ${index + 1}</h3>
      <p class="meta"><strong class="${identityClass}">${displayIdentity}</strong></p>
      <p class="meta">Confidence: ${Number(match.confidence).toFixed(4)}</p>
      <p class="meta">Bounding box: [${match.box.join(", ")}]</p>
    `;

    if (!match.is_known) {
      const form = document.createElement("form");
      form.className = "identify-form";
      form.innerHTML = `
        <input type="text" name="name" placeholder="Enter identity name" required />
        <button type="submit">Save Identity</button>
      `;
      form.addEventListener("submit", (event) => submitIdentity(event, match));
      card.appendChild(form);
    }

    results.appendChild(card);
  });
}

async function submitIdentity(event, match) {
  event.preventDefault();

  if (!selectedFile || !imageBitmap) {
    setStatus("Please select an image first.");
    return;
  }

  const form = event.currentTarget;
  const button = form.querySelector("button");
  const name = new FormData(form).get("name")?.toString().trim();
  if (!name) {
    setStatus("A name is required.");
    return;
  }

  button.disabled = true;
  setStatus(`Saving identity for ${name}...`);

  try {
    const blob = await cropFace(match.box);
    const payload = new FormData();
    payload.append("name", name);
    payload.append("confidence", String(match.confidence));
    payload.append("file", blob, "crop.jpg");

    const response = await fetch(`${apiBaseUrl}/identify-face`, {
      method: "POST",
      body: payload,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Failed to save face.");
    }

    setStatus(`${body.name} saved. Re-running recognition...`);
    await rerunRecognition();
  } catch (error) {
    setStatus(error.message);
  } finally {
    button.disabled = false;
  }
}

async function cropFace(box) {
  const [x1, y1, x2, y2] = box;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageBitmap, x1, y1, width, height, 0, 0, width, height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error("failed to create crop"));
    }, "image/jpeg", 0.95);
  });
}

async function rerunRecognition() {
  const formData = new FormData();
  formData.append("file", selectedFile);
  const response = await fetch(`${apiBaseUrl}/recognize`, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Recognition failed.");
  }
  currentMatches = payload.matches || [];
  drawMatches(currentMatches);
  renderResults();
  setStatus(`Recognition complete. ${currentMatches.length} face(s) detected.`);
}

function resetPreview() {
  previewShell.classList.add("empty");
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  imageBitmap = null;
  clearOverlay();
}

function resetSelection() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  selectedFile = null;
  fileInput.value = "";
  resetPreview();
}

function isSupportedFile(file) {
  const name = file.name.toLowerCase();
  return supportedExtensions.some((extension) => name.endsWith(extension));
}

function setStatus(message) {
  statusText.textContent = message;
}

function resolveApiBaseUrl() {
  const configuredUrl =
    window.__FACE_API_BASE_URL__ ||
    document.querySelector('meta[name="face-api-base-url"]')?.content ||
    "";
  return configuredUrl.replace(/\/$/, "") || window.location.origin;
}
