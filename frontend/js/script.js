// Configuration
const API_URL = "http://localhost:8000";
const GRID_SIZE = 28;
const CANVAS_SIZE = 280;
const PIXEL_SIZE = CANVAS_SIZE / GRID_SIZE; // 10px per grid cell

// Canvas setup
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

// Grid data: 28x28 array storing intensities [0, 255]
let grid = Array(GRID_SIZE)
  .fill(0)
  .map(() => Array(GRID_SIZE).fill(0));

// Drawing state
let isDrawing = false;
let lastX = -1;
let lastY = -1;

// Initialize canvas
function initCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

// Convert canvas coordinates to grid coordinates
function getGridCoords(canvasX, canvasY) {
  const x = Math.floor(canvasX / PIXEL_SIZE);
  const y = Math.floor(canvasY / PIXEL_SIZE);
  return {
    x: Math.max(0, Math.min(GRID_SIZE - 1, x)),
    y: Math.max(0, Math.min(GRID_SIZE - 1, y)),
  };
}

// Draw on the grid with smooth Gaussian brush
function drawAtPosition(canvasX, canvasY) {
  const { x, y } = getGridCoords(canvasX, canvasY);

  // Gaussian brush with smooth falloff (thinner stroke)
  const sigma = 0.5; // Smaller = thinner stroke

  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      const nx = x + dx;
      const ny = y + dy;

      if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
        // Gaussian intensity: exp(-(distance^2) / (2*sigma^2))
        const distance = Math.sqrt(dx * dx + dy * dy);
        const intensity = Math.exp(
          -(distance * distance) / (2 * sigma * sigma),
        );

        // Scale to [0, 255] with max intensity at center
        const addedIntensity = intensity * 100; // Reduced from 255 for softer stroke

        // Additive blending (don't overwrite, add up to max 255)
        grid[ny][nx] = Math.min(255, grid[ny][nx] + addedIntensity);

        // Render on canvas
        const value = grid[ny][nx];
        ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
        ctx.fillRect(nx * PIXEL_SIZE, ny * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
      }
    }
  }

  updatePixelCount();
}

// Interpolate drawing between two points (for smooth lines)
function drawLine(x0, y0, x1, y1) {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;

  while (true) {
    drawAtPosition(x0, y0);

    if (x0 === x1 && y0 === y1) break;

    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0 += sy;
    }
  }
}

// Event Handlers
canvas.addEventListener("mousedown", (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  drawAtPosition(x, y);
  lastX = x;
  lastY = y;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDrawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  // Draw smooth line from last position to current
  if (lastX !== -1 && lastY !== -1) {
    drawLine(lastX, lastY, x, y);
  }

  lastX = x;
  lastY = y;
});

canvas.addEventListener("mouseup", () => {
  isDrawing = false;
  lastX = -1;
  lastY = -1;
});

canvas.addEventListener("mouseleave", () => {
  isDrawing = false;
  lastX = -1;
  lastY = -1;
});

// Touch support for mobile
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  drawAtPosition(x, y);
  lastX = x;
  lastY = y;
});

canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  if (!isDrawing) return;

  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  if (lastX !== -1 && lastY !== -1) {
    drawLine(lastX, lastY, x, y);
  }

  lastX = x;
  lastY = y;
});

canvas.addEventListener("touchend", () => {
  isDrawing = false;
  lastX = -1;
  lastY = -1;
});

// Clear button
document.getElementById("clearButton").addEventListener("click", () => {
  grid = Array(GRID_SIZE)
    .fill(0)
    .map(() => Array(GRID_SIZE).fill(0));
  initCanvas();
  updatePixelCount();
  showWaitingState();
});

// Predict button
document.getElementById("predictButton").addEventListener("click", async () => {
  await makePrediction();
});

// Update pixel count
function updatePixelCount() {
  let count = 0;
  for (let row of grid) {
    for (let val of row) {
      if (val > 0) count++;
    }
  }
  document.getElementById("pixelCount").textContent = count;
}

// Calculate center of mass of the drawing
function getCenterOfMass() {
  let totalMass = 0;
  let xSum = 0;
  let ySum = 0;

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const intensity = grid[row][col];
      if (intensity > 0) {
        totalMass += intensity;
        xSum += col * intensity;
        ySum += row * intensity;
      }
    }
  }

  if (totalMass === 0) {
    return { x: GRID_SIZE / 2, y: GRID_SIZE / 2 };
  }

  return {
    x: xSum / totalMass,
    y: ySum / totalMass,
  };
}

// Center the drawing in the grid (like MNIST preprocessing)
function centerGrid() {
  const com = getCenterOfMass();
  const centerX = GRID_SIZE / 2;
  const centerY = GRID_SIZE / 2;
  const shiftX = Math.round(centerX - com.x);
  const shiftY = Math.round(centerY - com.y);

  // Create new centered grid
  const centeredGrid = Array(GRID_SIZE)
    .fill(0)
    .map(() => Array(GRID_SIZE).fill(0));

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const newRow = row + shiftY;
      const newCol = col + shiftX;

      if (
        newRow >= 0 &&
        newRow < GRID_SIZE &&
        newCol >= 0 &&
        newCol < GRID_SIZE
      ) {
        centeredGrid[newRow][newCol] = grid[row][col];
      }
    }
  }

  return centeredGrid;
}

// Convert grid to flat array for API (ROW-MAJOR ORDER, same as MNIST)
function getPixelArray() {
  // Center the image first (like MNIST preprocessing)
  const centeredGrid = centerGrid();

  const pixels = [];

  // IMPORTANT: Read row by row, left to right (row-major order)
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const intensity = centeredGrid[row][col];
      pixels.push(intensity / 255.0); // Normalize to [0, 1]
    }
  }

  return pixels;
}

// Make prediction
async function makePrediction() {
  const pixels = getPixelArray();

  // Check if drawing is empty
  const isEmpty = pixels.every((p) => p === 0);
  if (isEmpty) {
    showError("Please draw a digit first!");
    return;
  }

  // Debug log
  console.log("=== PREDICTION DEBUG ===");
  console.log("Grid dimensions:", GRID_SIZE, "x", GRID_SIZE);
  console.log("Total pixels:", pixels.length);
  console.log("Expected pixels:", 784);
  console.log("Non-zero pixels:", pixels.filter((p) => p > 0).length);
  console.log("Min value:", Math.min(...pixels));
  console.log("Max value:", Math.max(...pixels));
  console.log("First row (28 pixels):", pixels.slice(0, 28));
  console.log("========================");

  // Verify pixel count
  if (pixels.length !== 784) {
    showError(`Invalid pixel count: ${pixels.length} (expected 784)`);
    return;
  }

  try {
    showLoading();

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ pixels }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log("Prediction result:", data);
    showPrediction(data);
  } catch (error) {
    console.error("Prediction error:", error);
    showError(`Failed to predict: ${error.message}`);
  }
}

// UI States
function showWaitingState() {
  const content = document.getElementById("resultsContent");
  content.innerHTML = `
    <div class="waiting-state">
      <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
      </svg>
      <p>Draw a digit to see prediction</p>
    </div>
  `;
}

function showLoading() {
  const content = document.getElementById("resultsContent");
  content.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <p class="loading-text">Analyzing your drawing...</p>
    </div>
  `;
}

function showError(message) {
  const content = document.getElementById("resultsContent");
  content.innerHTML = `
    <div class="error-state">
      <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <p class="error-message">${message}</p>
    </div>
  `;
}

function showPrediction(data) {
  const { prediction, probabilities, confidence } = data;

  // Generate probability bars
  let probabilityHTML = "";
  for (let i = 0; i < 10; i++) {
    const prob = probabilities[i.toString()];
    const percentage = (prob * 100).toFixed(1);
    const isMax = i === prediction;

    probabilityHTML += `
      <div class="probability-row ${isMax ? "max" : ""}">
        <span class="probability-digit">${i}</span>
        <div class="probability-bar-container">
          <div class="probability-bar" style="width: ${percentage}%"></div>
        </div>
        <span class="probability-value">${percentage}%</span>
      </div>
    `;
  }

  const content = document.getElementById("resultsContent");
  content.innerHTML = `
    <div class="prediction-display">
      <div class="prediction-main">
        <div class="prediction-digit">${prediction}</div>
        <div class="prediction-info">
          <div class="prediction-confidence">${(confidence * 100).toFixed(1)}%</div>
          <div class="prediction-label">Confidence</div>
        </div>
      </div>
      <div class="probabilities-section">
        <div class="probabilities-title">All Probabilities</div>
        ${probabilityHTML}
      </div>
    </div>
  `;
}

// Test API connection
async function testConnection() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (response.ok) {
      const data = await response.json();
      console.log("✓ API connected:", data);
    }
  } catch (error) {
    console.warn(
      "✗ API not available. Start backend with: cd backend && python3 main.py",
    );
  }
}

// Initialize
initCanvas();
testConnection();
