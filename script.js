const settingsBtn = document.getElementById("settingsBtn");
const settingsPopup = document.getElementById("settingsPopup");
const updateSlider = document.getElementById("updateSlider");
const percentDisplay = document.getElementById("percentDisplay");

function updatePercentage() {
  const value = parseFloat(updateSlider.value).toFixed(2);
  percentDisplay.textContent = `${value}%`;
}

settingsBtn.addEventListener("click", () => {
  settingsPopup.classList.toggle("hidden");
});

updateSlider.addEventListener("input", updatePercentage);

document.addEventListener("click", (event) => {
  const clickedInsidePopup = settingsPopup.contains(event.target);
  const clickedGear = settingsBtn.contains(event.target);

  if (!clickedInsidePopup && !clickedGear) {
    settingsPopup.classList.add("hidden");
  }
});

updatePercentage();