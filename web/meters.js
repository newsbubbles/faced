/* Renders the emotion meter strip and updates bar widths from the SSE stream. */
(function () {
  let rows = {};

  function init(container, meta) {
    container.innerHTML = "";
    rows = {};
    meta.emotions.forEach(e => {
      const ax = meta.axes[e] || {};
      const div = document.createElement("div");
      div.className = "m" + (ax.accepted ? "" : " weak");
      const label = ax.bipolar && ax.neg_label ? `${ax.neg_label} ↔ ${e}` : e;
      div.innerHTML =
        `<div class="lab">${label}</div>` +
        `<div class="track">${ax.bipolar ? '<div class="mid"></div>' : ''}` +
        `<div class="fill${ax.bipolar ? ' bip' : ''}"></div></div>` +
        `<div class="val">–</div>`;
      container.appendChild(div);
      rows[e] = { fill: div.querySelector(".fill"), val: div.querySelector(".val"),
                  bipolar: ax.bipolar };
    });
  }

  function update(meters) {
    for (const e in meters) {
      const r = rows[e];
      if (!r) continue;
      const v = meters[e].value;
      if (r.bipolar) {
        // fill from center outward
        const dev = Math.abs(v - 50);
        r.fill.style.left = (v >= 50 ? 50 : 50 - dev) + "%";
        r.fill.style.width = dev + "%";
      } else {
        r.fill.style.left = "0%";
        r.fill.style.width = v + "%";
      }
      r.val.textContent = v.toFixed(0);
    }
  }

  window.Meters = { init, update };
})();
