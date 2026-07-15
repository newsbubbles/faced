/* SVG face driven by FACS-lite params in [-1,1]. Eases toward targets each frame
   so emotion changes read as microexpressions. Adds idle blink + gaze saccades. */
(function () {
  const G = {
    eye: { L: { x: 150, y: 182 }, R: { x: 270, y: 182 }, rx: 34, ry: 24 },
    brow: { baseY: 140,
            L: { inner: 186, outer: 116 }, R: { inner: 234, outer: 304 } },
    mouth: { cx: 210, cy: 302, xL: 164, xR: 256 },
  };
  const KEYS = ["brow_inner","brow_outer","brow_furrow","lid_upper","lid_lower_tense",
                "gaze_x","gaze_y","mouth_curve","mouth_open","cheek_raise",
                "head_tilt","head_pitch","asymmetry"];
  // per-param easing: brows/eyes snappy, mouth medium, head slow.
  const EASE = { brow_inner:.28, brow_outer:.28, brow_furrow:.28, lid_upper:.35,
                 lid_lower_tense:.3, gaze_x:.2, gaze_y:.2, mouth_curve:.2,
                 mouth_open:.24, cheek_raise:.15, head_tilt:.08, head_pitch:.08, asymmetry:.2 };

  let el = {}, cur = {}, tgt = {}, active = false, blink = 0, sacc = { x:0, y:0 }, tSacc = 0;

  function $(id) { return document.getElementById(id); }

  function init(svg) {
    ["head","browL","browR","eyeWhiteL","eyeWhiteR","pupilL","pupilR",
     "lidUpL","lidUpR","lidLoL","lidLoR","mouth","cheekL","cheekR"]
      .forEach(id => el[id] = svg.querySelector("#" + id));
    KEYS.forEach(k => { cur[k] = 0; tgt[k] = 0; });
    // setInterval (not requestAnimationFrame) so easing runs even when the pane
    // isn't actively painting / is backgrounded.
    setInterval(() => frame(performance.now()), 33);
    frame(performance.now());
  }

  function target(p) {
    KEYS.forEach(k => { if (k in p) tgt[k] = p[k]; });
    frame(performance.now());  // immediate step toward the new target per token
  }
  function setActive(a) { active = a; }

  function browPath(side) {
    const b = G.brow[side], base = G.brow.baseY;
    const asym = (side === "L" ? 1 : -1) * cur.asymmetry * 7;
    const innerY = base - cur.brow_inner*16 + cur.brow_furrow*11 + asym;
    const outerY = base - cur.brow_outer*15;
    const innerX = b.inner + (side === "L" ? 1 : -1) * cur.brow_furrow*7;
    const midX = (innerX + b.outer) / 2;
    const midY = Math.min(innerY, outerY) - 7 - cur.brow_outer*3;
    return `M ${b.outer} ${outerY} Q ${midX} ${midY} ${innerX} ${innerY}`;
  }

  function lids(side) {
    const e = G.eye[side], rx = G.eye.rx, ry = G.eye.ry;
    const top = e.y - ry, bot = e.y + ry, x0 = e.x - rx - 2, x1 = e.x + rx + 2;
    // upper aperture: openU in ~[0.05,1.5]; blink forces closed
    let openU = 1 + cur.lid_upper;
    openU *= (1 - blink);
    openU = Math.max(0.02, Math.min(1.5, openU));
    const cover = Math.max(0, 1 - openU);           // fraction of eye covered from top
    const lidY = top + cover * (2 * ry);
    el["lidUp" + side].setAttribute("d",
      `M ${x0} ${top-34} L ${x1} ${top-34} L ${x1} ${lidY} Q ${e.x} ${lidY+9} ${x0} ${lidY} Z`);
    // lower lid tension raises from bottom
    const lo = Math.max(0, cur.lid_lower_tense) * 0.55;
    const loY = bot - lo * (2 * ry);
    el["lidLo" + side].setAttribute("d",
      `M ${x0} ${bot+34} L ${x1} ${bot+34} L ${x1} ${loY} Q ${e.x} ${loY-7} ${x0} ${loY} Z`);
  }

  function mouthPath() {
    const m = G.mouth;
    const cornerY = m.cy - cur.mouth_curve*10;
    const open = Math.max(0, cur.mouth_open);
    const topMidY = m.cy - cur.mouth_curve*17 - open*6;
    const botMidY = m.cy - cur.mouth_curve*17 + open*30 + 3;
    return `M ${m.xL} ${cornerY} Q ${m.cx} ${topMidY} ${m.xR} ${cornerY} `
         + `Q ${m.cx} ${botMidY} ${m.xL} ${cornerY} Z`;
  }

  function frame(ts) {
    // idle blink
    if (ts > blinkNext) { blinkT = ts; blinkNext = ts + 2600 + Math.random()*3800; }
    const bt = ts - blinkT;
    blink = bt < 160 ? Math.sin(Math.min(bt,160)/160 * Math.PI) : 0;
    // idle saccades when not generating
    if (!active && ts > tSacc) { sacc = { x:(Math.random()-.5)*0.5, y:(Math.random()-.5)*0.4 }; tSacc = ts + 900 + Math.random()*1600; }
    if (active) sacc = { x: sacc.x*0.9, y: sacc.y*0.9 };

    KEYS.forEach(k => { cur[k] += (tgt[k] - cur[k]) * EASE[k]; });

    el.head.setAttribute("transform",
      `rotate(${cur.head_tilt*8} 210 210) translate(0 ${-cur.head_pitch*10}) `
      + `scale(${1 + cur.head_pitch*0.03})`);
    el.browL.setAttribute("d", browPath("L"));
    el.browR.setAttribute("d", browPath("R"));
    lids("L"); lids("R");
    ["L","R"].forEach(s => {
      const e = G.eye[s];
      el["pupil"+s].setAttribute("cx", e.x + (cur.gaze_x+sacc.x)*11);
      el["pupil"+s].setAttribute("cy", e.y + (cur.gaze_y+sacc.y)*8);
    });
    el.mouth.setAttribute("d", mouthPath());
    const ch = Math.max(0, cur.cheek_raise);
    el.cheekL.setAttribute("opacity", ch*0.5);
    el.cheekR.setAttribute("opacity", ch*0.5);
  }

  let blinkNext = 1500, blinkT = -9999;
  window.Face = { init, target, setActive };
})();
