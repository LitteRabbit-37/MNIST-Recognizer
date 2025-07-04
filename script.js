const container = document.getElementById("container");
const colors = ["#e74c3c", "#8e44ad", "#3498db", "#e67e22", "#2ecc71"];
const SQUARES = 28 * 28;

let isLeftMouseDown = false;

// Track Left mouse button state
document.addEventListener("contextmenu", (e) => e.preventDefault()); // Prevent context menu
document.addEventListener("mousedown", (e) => {
    if (e.button === 0) isLeftMouseDown = true;
});
document.addEventListener("mouseup", (e) => {
    if (e.button === 0) isLeftMouseDown = false;
});

// Reset squares when reset button is clicked (id = resetButton)
document.getElementById("resetButton").addEventListener("click", () => {
    const squares = document.querySelectorAll(".square");
    squares.forEach((square) => {
        square.style.background = "#333";
        square.style.border = "1px solid #111";
        square.style.boxShadow = "none";
    });
});

for (let i = 0; i < SQUARES; i++) {
    const square = document.createElement("div");
    square.classList.add("square");
    square.addEventListener("mouseover", () => {
        if (isLeftMouseDown) {
            setColor(square);
        }
    });
    // remove color on right click
    square.addEventListener("mousedown", (e) => {
        if (e.button === 2) {
            removeColor(square);
        }
    });
    container.appendChild(square);
}

function getRandomColor() {
    return colors[Math.floor(Math.random() * colors.length)];
}

function setColor(element) {
    const color = getRandomColor();
    element.style.background = color;
    element.style.boxShadow = `0 0 2px ${color}, 0 0 10px ${color}`;
}

function removeColor(element) {
    element.style.background = "#333";
    element.style.boxShadow = "none";
}
